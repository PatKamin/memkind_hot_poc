
#include <memkind/internal/pebs.h>
#include <memkind/internal/tachanka.h>
#include <memkind/internal/critnib.h>
#include <memkind/internal/memkind_private.h>
#include <memkind/internal/memkind_memtier.h>
#include <memkind/internal/memkind_log.h>

#include <assert.h>

#define rmb() asm volatile("lfence":::"memory")

typedef enum {
    THREAD_INIT,
    THREAD_RUNNING,
    THREAD_FINISHED,
} ThreadState_t;

pthread_t pebs_thread;
ThreadState_t thread_state = THREAD_INIT;
int pebs_fd;
static char *pebs_mmap;

#if CHECK_ADDED_SIZE
extern size_t g_total_ranking_size;
extern size_t g_total_critnib_size;
#endif

extern critnib* hash_to_type;
static size_t g_queue_pop_counter=0;
static size_t g_queue_counter_malloc=0;
static size_t g_queue_counter_realloc=0;
static size_t g_queue_counter_callback=0;
static size_t g_queue_counter_free=0;
static size_t g_queue_counter_touch=0;
extern struct ttype ttypes[];

// static uint64_t timespec_diff_millis(const struct timespec *tnew, const struct timespec *told) {
//     uint64_t diff_s = tnew->tv_sec - told->tv_sec;
//     uint64_t tnew_ns = tnew->tv_nsec;
//     uint64_t told_ns = told->tv_nsec;
//     if (told_ns > tnew_ns) {
//         take 1s and convert it to ns
//         const uint64_t s_to_ns=(uint64_t)1000000000;
//         tnew_ns += s_to_ns;
//         diff_s --;
//     }
//     return diff_s*1000+(tnew_ns-told_ns)/1000000;
// }

static void timespec_add(struct timespec *modified, const struct timespec *toadd) {
    const uint64_t S_TO_NS=1000000000u;
    modified->tv_sec += toadd->tv_sec;
    modified->tv_nsec += toadd->tv_nsec;
    modified->tv_sec += (modified->tv_nsec/S_TO_NS);
    modified->tv_nsec %= S_TO_NS;
}

static void timespec_millis_to_timespec(double millis, struct timespec *out) {
    out->tv_sec = (uint64_t)(millis/1000u);
    double ms_to_convert = millis-out->tv_sec*1000;
    out->tv_nsec = (uint64_t)(ms_to_convert*1000000);
}

static bool timespec_is_he(struct timespec *tv1, struct timespec *tv2) {
    return tv1->tv_sec>tv2->tv_sec || (tv1->tv_sec == tv2->tv_sec && tv1->tv_nsec > tv2->tv_nsec);
}

#if PEBS_LOG_TO_FILE
// DEBUG
static char *bp;
static int display_hotness(int nt)
{
    const struct ttype *t = &ttypes[nt];
    if (t->timestamp_state == TIMESTAMP_INIT_DONE)
        bp += sprintf(bp, "%f,", t->f);
    else
        bp += sprintf(bp, "N/A,");
    return 0;
}
#endif

void *pebs_monitor(void *state)
{
    ThreadState_t* pthread_state = state;

    double period_ms = 1000 / pebs_freq_hz;
    struct timespec tv_period;
    timespec_millis_to_timespec(period_ms, &tv_period);

    // set low priority
    int policy;
    struct sched_param param;
    pthread_getschedparam(pthread_self(), &policy, &param);
    param.sched_priority = sched_get_priority_min(policy);
    pthread_setschedparam(pthread_self(), policy, &param);

    __u64 last_head = 0;
#if PRINT_PEBS_BASIC_INFO
    int cur_tid = syscall(SYS_gettid);
#endif

    // DEBUG
#if PEBS_LOG_TO_FILE
    char buf[4*1048576];
    static int pid;
    static int log_file;
    int cur_pid = getpid();
    printf("starting pebs monitor for thread %d", cur_tid);
    if (pid != cur_pid) {
        char name[255] = {0};
        sprintf(name, "tier_pid_%d_tid_%d.log", cur_pid, cur_tid);
        log_file = open(name, O_CREAT | O_WRONLY, S_IRWXU);
    }
    lseek(log_file, 0, SEEK_END);
#endif

    struct timespec ntime;
    int ret = clock_gettime(CLOCK_MONOTONIC, &ntime);
    if (ret != 0) {
        log_fatal("ASSERT PEBS Monitor CLOCK_GETTIME_FAILURE!");
        exit(-1);
    }
    timespec_add(&ntime, &tv_period);

    while (1) {
        // TODO - use mutex?
        if (*pthread_state == THREAD_FINISHED) {
            return NULL;
        }

        // must call this before read from data head
		rmb();

#if PRINT_PEBS_STATS_ON_COUNTER_OVERFLOW_INFO
        {
            static uint64_t counter = 0;
            const uint64_t interval = 1000;
            if (++counter > interval) {
                struct timespec t;
                int ret = clock_gettime(CLOCK_MONOTONIC, &t);
                if (ret != 0) {
                    log_fatal("ASSERT_CLOCK_GETTIME_FAILURE!\n");
                    exit(-1);
                }

                log_info("pebs counter %lu hit, succcessful ranking_queue reads %lu, "
                    "time [seconds, nanoseconds]: [%ld, %ld]",
                    interval, g_queue_pop_counter, t.tv_sec, t.tv_nsec);
                log_info("g_queue_counter_malloc: %lu, g_queue_counter_realloc: %lu, "
                    "g_queue_counter_callback: %lu, g_queue_counter_free: %lu, g_queue_counter_touch: %lu",
                    g_queue_counter_malloc, g_queue_counter_realloc, g_queue_counter_callback,
                    g_queue_counter_free, g_queue_counter_touch);
                counter=0u;
            }
        }
#endif

        EventEntry_t event;
        bool pop_success;
        while (true) {
            pop_success = tachanka_ranking_event_pop(&event);
            if (!pop_success)
                break;
            switch (event.type) {
                case EVENT_CREATE_ADD: {
                    log_info("EVENT_CREATE_ADD");
                    EventDataCreateAdd *data = &event.data.createAddData;
                    register_block(data->hash, data->address, data->size);
                    register_block_in_ranking(data->address, data->size);
                    g_queue_counter_malloc++;
                    break;
                }
                case EVENT_DESTROY_REMOVE: {
                    log_info("EVENT_DESTROY_REMOVE");
                    EventDataDestroyRemove *data = &event.data.destroyRemoveData;
                    // REMOVE THE BLOCK FROM RANKING!!!
                    // TODO remove all the exclamation marks and clean up once this is done
                    unregister_block(data->address);
                    g_queue_counter_free++;
                    break;
                }
                case EVENT_REALLOC: {
                    log_info("EVENT_REALLOC");
                    EventDataRealloc *data = &event.data.reallocData;
                    unregister_block(data->addressOld);
//                     realloc_block(data->addressOld, data->addressNew, data->sizeNew);
                    register_block(0u /* FIXME hash should not be zero !!! */, data->addressNew, data->sizeNew);
                    register_block_in_ranking(data->addressNew, data->sizeNew);
                    g_queue_counter_realloc++;
                    break;
                }
                case EVENT_SET_TOUCH_CALLBACK: {
                    log_info("EVENT_SET_TOUCH_CALLBACK");
                    EventDataSetTouchCallback *data = &event.data.touchCallbackData;
                    tachanka_set_touch_callback(data->address,
                                                data->callback,
                                                data->callbackArg);
                    g_queue_counter_callback++;
                    break;
                }
                /*
                case EVENT_TOUCH: {
                    int fromMalloc = 0; // false
                    EventDataTouch *data = &event.data.touchData;
                    touch(data->address, data->timestamp, fromMalloc);
                    g_queue_counter_touch++;
                    break;
                }*/
                default: {
                    log_fatal("PEBS: event queue - case not implemented!");
                    exit(-1);
                }
            }
            g_queue_pop_counter++;

#if CHECK_ADDED_SIZE
        log_info("EVENT end g_total_ranking_size %ld g_total_critnib_size %ld", 
            g_total_ranking_size, g_total_critnib_size);
#endif
        }

        struct perf_event_mmap_page* pebs_metadata =
            (struct perf_event_mmap_page*)pebs_mmap;

        if (last_head < pebs_metadata->data_head) {
#if PRINT_PEBS_NEW_DATA_INFO
            log_info("PEBS: new data to process!");
#endif
            int samples = 0;
            __u64 timestamp = *(__u64*)(pebs_mmap + getpagesize() +
                    last_head % (MMAP_DATA_SIZE * getpagesize()));

            while (last_head < pebs_metadata->data_head) {
	            char *data_mmap = pebs_mmap + getpagesize() +
                    last_head % (MMAP_DATA_SIZE * getpagesize());

                struct perf_event_header *event =
                    (struct perf_event_header *)data_mmap;

//  {    static uint64_t counter=0;
//     const uint64_t interval=10;
//     if (++counter > interval) {
//         struct timespec t;
//         int ret = clock_gettime(CLOCK_MONOTONIC, &t);
//         if (ret != 0) {
//             printf("ASSERT PEBS EVENT COUNTER FAILURE!\n");
//         }
//         assert(ret == 0);
//         printf("pebs event counter 10 hit, [seconds, nanoseconds]: [%ld, %ld]\n",
//             t.tv_sec, t.tv_nsec);
//         counter=0u;
//     }
// }
                switch (event->type) {
                    case PERF_RECORD_SAMPLE:
                    {
                        timestamp = *(__u64*)(data_mmap + sizeof(struct perf_event_header));
                        // 'addr' is the acessed address
                        __u64 addr = *(__u64*)(data_mmap + sizeof(struct perf_event_header) + sizeof(__u64));

                        // TODO - is this a global or per-core timestamp?
                        // If per-core, this could lead to some problems

// {    static uint64_t counter=0;
//     const uint64_t interval=10;
//     if (++counter > interval) {
//         struct timespec t;
//         int ret = clock_gettime(CLOCK_MONOTONIC, &t);
//         if (ret != 0) {
//             printf("ASSERT PEBS TOUCH COUNTER FAILURE!\n");
//         }
//         assert(ret == 0);
//         printf("pebs touch counter 10 hit, [seconds, nanoseconds]: [%ld, %ld]\n",
//             t.tv_sec, t.tv_nsec);
//         counter=0u;
//     }
// }
//                         printf("touches, timestamp: [%llu], from malloc [0]\n", timestamp);

                        // defer touch after corresponding malloc - put it onto queue
                        EventEntry_t entry = {
                            .type = EVENT_TOUCH,
                            .data.touchData = {
                                .address = (void*)addr,
                                .timestamp = timestamp,
                            },
                        };

                        // copy is performed, passing pointer to stack is ok
                        // losing single malloc should not cause issues,
                        // so we are just ignoring buffer overflows
//                         (void)tachanka_ranking_event_push(&entry); // TODO
                        touch(entry.data.touchData.address,
                            entry.data.touchData.timestamp, 0 /*called from malloc*/);
                            g_queue_counter_touch++;

//                         touch((void*)addr, timestamp, 0 /* from malloc */);
//                         printf("touched, timestamp: [%llu], from malloc [0]\n", timestamp);

#if PEBS_LOG_TO_FILE
                        // DEBUG
                        sprintf(buf, "last: %llu, head: %llu t: %llu addr: %llx\n",
                            last_head, pebs_metadata->data_head,
                            timestamp, addr);
                        //if (write(log_file, buf, strlen(buf))) ;
#endif

#if PRINT_PEBS_TOUCH_INFO
                        log_info("PEBS touch(): last: %llu, head: %llu t: %llu addr: %llx",
                            last_head, pebs_metadata->data_head,
                            timestamp, addr);
#endif
                    }
                    break;
                default:
                    break;
                };

                last_head += event->size;
                data_mmap += event->size;
                samples++;
            }

#if RANKING_TOUCH_ALL
            // Touch every ttype object to update hotness
            tachanka_ranking_touch_all(timestamp, 0);
#endif

#if PRINT_PEBS_SAMPLES_NUM_INFO
            log_info("PEBS: processed %d samples", samples);
#endif

#if PEBS_LOG_TO_FILE
            bp = buf;
            critnib_iter(hash_to_type, display_hotness);
            bp += sprintf(bp, "\n");
            if (write(log_file, buf, bp - buf));
#endif
        }

		ioctl(pebs_fd, PERF_EVENT_IOC_REFRESH, 0);
        last_head = pebs_metadata->data_head;
        pebs_metadata->data_tail = pebs_metadata->data_head;
        tachanka_update_threshold();

//         ret = clock_gettime(CLOCK_MONOTONIC, &ctime);
//         if (ret != 0) {
//             printf("ASSERT_CLOCK_GETTIME_FAILURE!\n");
//             assert(ret!=0);
//         }
//         uint64_t diff_ms = timespec_diff_millis(&ctime, &ptime);
//         if ()
//
//         usleep(1000*(diff_ms-period_ms))
//         if (diff_ms > period_ms) {
//             warning:
//         }
//         ptime = ctime;

// 		sleep(1); // TODO analysis depends on event number; better solution: const 1 Hz, with low priority
//         usleep(1000);

        struct timespec temp;
        ret = clock_gettime(CLOCK_MONOTONIC, &temp);
        if (ret != 0) {
            log_fatal("ASSERT_CLOCK_GETTIME_FAILURE!");
            exit(-1);
        }
#if PRINT_PEBS_TIMESPEC_DEADLINE_INFO
        if (timespec_is_he(&temp, &ntime)) {
            log_info("PEBS: timespec deadline not met!");
        }
#endif
        (void)clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ntime, NULL);
        timespec_add(&ntime, &tv_period);
    }

#if PRINT_PEBS_BASIC_INFO
    log_info("PEBS: stopping pebs monitor for thread %d", cur_tid);
#endif

    return NULL;
}

void pebs_init(pid_t pid)
{
    // TODO add code that writes to /proc/sys/kernel/perf_event_paranoid ?

#if PRINT_PEBS_BASIC_INFO
    log_info("PEBS: init");
#endif

    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(struct perf_event_attr));

    // NOTE: code bellow requires link to libpfm
    int ret = pfm_initialize();
    if (ret != PFM_SUCCESS) {
        log_fatal("PEBS: pfm_initialize() failed!");
        exit(-1);
    }

    pfm_perf_encode_arg_t arg;
    memset(&arg, 0, sizeof(arg));
    arg.attr = &pe;

    char event[] = "MEM_LOAD_RETIRED:L3_MISS";
    //char event[] = "MEM_UOPS_RETIRED:ALL_LOADS";

    ret = pfm_get_os_event_encoding(event, PFM_PLM3, PFM_OS_PERF_EVENT_EXT, &arg);
    if (ret != PFM_SUCCESS) {
        log_err("PEBS: pfm_get_os_event_encoding() failed - "
            "using magic numbers!");
        //exit(-1);

        pe.type = 4;
        pe.config = 0x5120D1;
    }

    pe.size = sizeof(struct perf_event_attr);
    pe.sample_period = sample_frequency;
    pe.sample_type = PERF_SAMPLE_ADDR | PERF_SAMPLE_TIME;

    pe.precise_ip = 2; // NOTE: this is reqired but was not set
                       // by pfm_get_os_event_encoding()
    pe.read_format = 0;
    pe.disabled = 1;
    pe.pinned = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    pe.wakeup_events = 1;

    // NOTE: pid is passed as an argument to this func
    //pid_t pid = 0;            // measure current process

    int cpu = -1;               // .. on any CPU
    int group_fd = -1;          // use single event group
    unsigned long flags = 0;
    pebs_fd = perf_event_open(&pe, pid, cpu, group_fd, flags);

    if (pebs_fd != -1) {
        int mmap_pages = 1 + MMAP_DATA_SIZE;
        int map_size = mmap_pages * getpagesize();
        pebs_mmap = mmap(NULL, map_size,
                        PROT_READ | PROT_WRITE, MAP_SHARED, pebs_fd, 0);

#if PRINT_PEBS_BASIC_INFO
        log_info("PEBS: thread start");
#endif

        thread_state = THREAD_RUNNING;
        pthread_create(&pebs_thread, NULL, &pebs_monitor, (void*)&thread_state);

        ioctl(pebs_fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(pebs_fd, PERF_EVENT_IOC_ENABLE, 0);
    }
    else
    {
        log_err("PEBS: PEBS NOT SUPPORTED! continuing without pebs!");
    }
}

void pebs_fini()
{
    // finish only if the thread is running
    if (thread_state == THREAD_RUNNING) {
        ioctl(pebs_fd, PERF_EVENT_IOC_DISABLE, 0);

        // TODO - use mutex?
        thread_state = THREAD_FINISHED;
        void* ret;
        pthread_join(pebs_thread, &ret);

        int mmap_pages = 1 + MMAP_DATA_SIZE;
        munmap(pebs_mmap, mmap_pages * getpagesize());
        pebs_mmap = 0;
        close(pebs_fd);

#if PRINT_PEBS_BASIC_INFO
        log_info("PEBS: thread end");
#endif
    }
}

MEMKIND_EXPORT void pebs_fork(pid_t pid)
{
#if PRINT_PEBS_BASIC_INFO
    log_info("PEBS: fork: %i", pid);
#endif
    pebs_init(pid);
}
