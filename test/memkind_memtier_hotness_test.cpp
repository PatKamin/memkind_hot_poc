// SPDX-License-Identifier: BSD-2-Clause
/* Copyright (C) 2021 Intel Corporation. */

#include <memkind/internal/memkind_memtier.h>
#include <memkind/internal/tachanka.h>
#include <memkind/internal/slab_allocator.h>
#include <memkind/internal/wre_avl_tree_internal.h>
#include <memkind/internal/ranking_fixer.h>

#include <random>
#include <thread>
#include <vector>
#include <mutex>
#include <pthread.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>

#include "common.h"
#include "zipf.h"

void naive_matrix_multiply(int matrix_size, int mul_step,
                                            double *a, double *b, double *c) {
    double s;
    int i,j,k;

    for(i=0;i<matrix_size;i++) {
        for(j=0;j<matrix_size;j+=mul_step) {
            a[i * matrix_size + j]=(double)i*(double)j;
            b[i * matrix_size + j]=(double)i/(double)(j+5);
        }
    }

    for(j=0;j<matrix_size;j++) {
        for(i=0;i<matrix_size;i++) {
            s=0;
            for(k=0;k<matrix_size;k+=mul_step) {
                s+=a[i * matrix_size + k]*b[k * matrix_size + j];
            }
            c[i * matrix_size + j] = s;
        }
    }

    s = 0.0;
    for(i=0;i<matrix_size;i++) {
        for(j=0;j<matrix_size;j+=mul_step) {
            s+=c[i * matrix_size + j];
        }
    }

    return;
}

class RandomIncremeneter {
    static std::random_device dev;
    std::mt19937 generator;
public:

    RandomIncremeneter() : generator(dev()) { }

    void IncrementRandom(volatile uint8_t *data, size_t size) {
        ASSERT_TRUE(data != NULL);
        std::uniform_int_distribution<uint64_t> distr(0, size-1);
        size_t index = distr(generator);
        ASSERT_TRUE(index < size);
        data[index]++;
    }
};

std::random_device RandomIncremeneter::dev;

class MemkindMemtierKindTest: public ::testing::Test
{
private:
    void SetUp()
    {}
    void TearDown()
    {}
};

class MemkindMemtierHotnessTest: public ::testing::Test,
      public ::testing::WithParamInterface<int>
{
protected:
    struct memtier_builder *m_builder;
    struct memtier_memory *m_tier_memory;

private:
    void SetUp()
    {
        m_tier_memory = nullptr;
        m_builder = memtier_builder_new(MEMTIER_POLICY_DATA_HOTNESS);
        ASSERT_NE(nullptr, m_builder);
    }

    void TearDown()
    {
        memtier_builder_delete(m_builder);
        if (m_tier_memory) {
            memtier_delete_memtier_memory(m_tier_memory);
        }
    }
};

TEST_F(MemkindMemtierHotnessTest, test_tier_two_kinds)
{
    int res = memtier_builder_add_tier(m_builder, MEMKIND_DEFAULT, 1);
    ASSERT_EQ(0, res);
    res = memtier_builder_add_tier(m_builder, MEMKIND_REGULAR, 1);
    ASSERT_EQ(0, res);
    m_tier_memory = memtier_builder_construct_memtier_memory(m_builder);
    ASSERT_NE(nullptr, m_tier_memory);
}

TEST_P(MemkindMemtierHotnessTest, test_matmul)
{
    const int MATRIX_SIZE = 512;
    const int MUL_STEP = 5;
    int OBJS_NUM = GetParam();

    // objects will be reallocated after N uses
    const int AGE_THRESHOLD = 10;
    const int LOOP_LEN = 20 * OBJS_NUM;
    // start iteration of hotness validation
    const int LOOP_CHECK_START = 5 * OBJS_NUM;
    // compare sum of hotness between objects from DEPTH num of checks
    const int LOOP_CHECK_DEPTH = 10;
    // get object hotness every FREQ iterations
    const int LOOP_CHECK_FREQ = 10;

    int it, it2;

    // setup only DRAM tier
    int res = memtier_builder_add_tier(m_builder, MEMKIND_DEFAULT, 1);
    ASSERT_EQ(0, res);
    // currently, adding only one tier is not supported
    // workaround: add two tiers, both dram
    res = memtier_builder_add_tier(m_builder, MEMKIND_REGULAR, 1);
    ASSERT_EQ(0, res);
    m_tier_memory = memtier_builder_construct_memtier_memory(m_builder);
    ASSERT_NE(nullptr, m_tier_memory);

    int mat_size = sizeof(double) * MATRIX_SIZE * MATRIX_SIZE;

    float accum_hotness[OBJS_NUM][LOOP_LEN] = {0};
    for (it = 0; it < OBJS_NUM; it++)
        for (it2 = 0; it2 < LOOP_LEN; it2++)
            accum_hotness[it][it2]=0;

    double** objs = (double**)malloc(OBJS_NUM * sizeof(double*));
    for (it = 0; it < OBJS_NUM; it++) {
        objs[it] = 0;
    }

    // fill frequency array using zipf distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    zipf_distribution<> zipf(OBJS_NUM);
    const int FREQ_ARRAY_LEN = LOOP_LEN;
    int freq[LOOP_LEN];
    for (it = 0; it < LOOP_LEN; it++) {
        freq[it] = zipf(gen) - 1;
    }

    // set age of each object to AGE_THRESHOLD to reallocate it immediately
    int ages[OBJS_NUM];
    for (it = 0; it < OBJS_NUM; it++) {
        ages[it] = AGE_THRESHOLD;
    }

    int sel = 0;
    int ready_to_validate = 0;
    int check_freq = 0;
	for (it = 0; it < LOOP_LEN; it++) {

        // select src1, src2 and dest objects
        sel++;
        sel = sel % FREQ_ARRAY_LEN;

        int dest_obj_id = freq[sel];
        double* dest_obj = objs[dest_obj_id];

        // each object has an age - if it goes above AGE_THRESHOLD
        // object is reallocated
        ages[dest_obj_id]++;
        if (ages[dest_obj_id] > AGE_THRESHOLD) {
            ages[dest_obj_id] = 0;

            memtier_free(dest_obj);
            objs[dest_obj_id] = (double*)memtier_malloc(m_tier_memory,
                mat_size + dest_obj_id * sizeof(double));
            dest_obj = objs[dest_obj_id];
            ASSERT_NE(nullptr, dest_obj);

            // DEBUG
            //printf("remalloc %d, start %llx, end %llx\n", dest_obj_id,
            //    (long long unsigned int)(&objs[dest_obj_id][0]),
            //    (long long unsigned int)(&objs[dest_obj_id][MATRIX_SIZE * MATRIX_SIZE - 1]));
        }

	    naive_matrix_multiply(MATRIX_SIZE, MUL_STEP,
            dest_obj, dest_obj, dest_obj);

        if (ready_to_validate == 0) {
            int num_allocated_objs = 0;
            for (int it2 = 0; it2 < OBJS_NUM; it2++) {
                if (objs[it2] != NULL)
                num_allocated_objs++;
            }

            if (num_allocated_objs == OBJS_NUM) {
                for (int it2 = 0; it2 < OBJS_NUM; it2++) {
                    accum_hotness[it2][it] = tachanka_get_obj_hotness(mat_size + it2 * sizeof(double));
                }

                if (it > LOOP_CHECK_START) {
                    ready_to_validate = 1;
                }
            }
        } else {
            check_freq ++;
            if (check_freq < LOOP_CHECK_FREQ)
                continue;

            check_freq = 0;
            for (int it2 = 1; it2 < OBJS_NUM; it2++) {
                float h0 = 0, h1 = 0;
                for (size_t it3 = 0; it3 < LOOP_CHECK_DEPTH; it3++) {
                    h0 += accum_hotness[it2 - 1][it - it3];
                    h1 += accum_hotness[it2][it - it3];
                }
                ASSERT_GE(h0, h1);
            }
        }

        // DEBUG
        //printf("dst %d\n", dest_obj_id);
        //fflush(stdout);

        // TODO check object hotness
        //
        // How the check for hotness should look like:
        //  1) sort all objects/types by their frequency
        //  2) calculate sum of all sizes
        //  3) hand-calculate which objects should be cold and which ones hot
        //  4) make a check in two loops: first loop for hot, second loop for cold
        //
        // Possible issues:
        //  1) pebs thread has not done its work; mitigation:
        //      a) wait (race condition-based mitigation)
        //      b) explicitly call pebs thread
        //  2) general race condition with pebs:
        //      i) cases:
        //          a) not enough: see point 1,
        //          b) too many times: old time window is gone,
        //          latest has 0 measurements
        //      ii) mitigation:
        //          a) explicitly call pebs, without separate thread
        //
        // For now, only "quickfix": make a test that's vulnerable to race condition
	}
}

INSTANTIATE_TEST_CASE_P(numObjsParam, MemkindMemtierHotnessTest,
                        ::testing::Values(3, 20));

TEST_F(MemkindMemtierHotnessTest, check_ranking_touch_all) {
    const int MALLOC_COUNT = 5;
    __u64 hotness_measure_window = 1000000000; // equals to the hotness_measure_window defined in pebs_init()
    __u64 wait_time = hotness_measure_window + 1;
    std::vector<void *> malloc_vec;

    int res = memtier_builder_add_tier(m_builder, MEMKIND_DEFAULT, 1);
    ASSERT_EQ(0, res);
    res = memtier_builder_add_tier(m_builder, MEMKIND_REGULAR, 1);
    ASSERT_EQ(0, res);
    m_tier_memory = memtier_builder_construct_memtier_memory(m_builder);
    ASSERT_NE(nullptr, m_tier_memory);

    for (size_t i = 1; i <= MALLOC_COUNT; ++i) {
        void *ptr = memtier_malloc(m_tier_memory, i);
        ASSERT_NE(nullptr, ptr);
        malloc_vec.push_back(ptr);
    }

    
    sleep(1); // wait for pebs_monitor() to register new types from memtier_mallocs

    for (size_t i = 0; i < MALLOC_COUNT; ++i) {
        ASSERT_EQ(tachanka_get_timestamp_state(i), TIMESTAMP_NOT_SET);
    }
    tachanka_ranking_touch_all(wait_time, 0); // set timestamp state to TIMESTAMP_INIT
    for (size_t i = 0; i < MALLOC_COUNT; ++i) {
        ASSERT_EQ(tachanka_get_frequency(i), 0);
    }
    for (size_t i = 0; i < MALLOC_COUNT; ++i) {
        ASSERT_EQ(tachanka_get_timestamp_state(i), TIMESTAMP_INIT);
    }

    tachanka_ranking_touch_all(2 * wait_time, 1e6); // set timestamp state to TIMESTAMP_INIT_DONE, set f to non-zero value
    for (size_t i = 0; i < MALLOC_COUNT; ++i) {
        ASSERT_EQ(tachanka_get_frequency(i), 0);
    }
    for (size_t i = 0; i < MALLOC_COUNT; ++i) {
        ASSERT_EQ(tachanka_get_timestamp_state(i), TIMESTAMP_INIT_DONE);
    }

    tachanka_ranking_touch_all(3 * wait_time, 0);  // touch all ttypes without adding the hotness
    for (size_t i = 0; i < MALLOC_COUNT; ++i) {
        ASSERT_GT(tachanka_get_frequency(i), 0);
    }

    for (auto const &ptr : malloc_vec) {
        memtier_free(ptr);
    }
}

// ----------------- hotness thresh tests

extern "C" {
#include "memkind/internal/ranking.h"
#include "memkind/internal/wre_avl_tree.h"
}

class RankingTest: public ::testing::Test
{
protected:
    ranking_t *ranking;
    static constexpr size_t BLOCKS_SIZE=100u;
    struct ttype blocks[BLOCKS_SIZE];
private:
    void SetUp()
    {
        ranking_create(&ranking, 0.9);

        for (size_t i=0; i<BLOCKS_SIZE; ++i) {
            blocks[i].num_allocs=BLOCKS_SIZE-i;
            blocks[i].f=i;
            ranking_add(ranking, blocks[i].f,  blocks[i].num_allocs);
        }

    }
    void TearDown()
    {
        ranking_destroy(ranking);
    }
};
constexpr size_t RankingTest::BLOCKS_SIZE;

double quantify_dequantify(double hotness) {
    return exp(int(log(hotness)));
}

TEST_F(RankingTest, check_hotness_highest) {
    double RATIO_PMEM_ONLY=0;
    double thresh_highest =
        ranking_calculate_hot_threshold_dram_total(
            ranking, RATIO_PMEM_ONLY, RATIO_PMEM_ONLY).threshVal;
    double thresh_highest_pmem =
        ranking_calculate_hot_threshold_dram_pmem(ranking, 0, 0).threshVal;
    ASSERT_EQ(thresh_highest, thresh_highest_pmem); // double for equality
#if QUANTIFICATION_ENABLED
    ASSERT_EQ(thresh_highest, quantify_dequantify(BLOCKS_SIZE-1));
    ASSERT_EQ(thresh_highest, quantify_dequantify(99));
    double ACCURACY = 1e-6;
    ASSERT_LE(abs(thresh_highest-54.598150033144236), ACCURACY);
    for (volatile size_t i=0; i<55; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
    // last one is hot due to quantification
    for (volatile size_t i=55; i<BLOCKS_SIZE; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[BLOCKS_SIZE-1]), true);
    }
#else
    ASSERT_EQ(thresh_highest, BLOCKS_SIZE-1);
    ASSERT_EQ(thresh_highest, 99);
    for (volatile size_t i=0; i<BLOCKS_SIZE; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
#endif
}

TEST_F(RankingTest, check_hotness_lowest) {
    double RATIO_DRAM_ONLY=1;
    double thresh_lowest =
        ranking_calculate_hot_threshold_dram_total(
            ranking, RATIO_DRAM_ONLY, RATIO_DRAM_ONLY).threshVal;
    double thresh_lowest_pmem =
        ranking_calculate_hot_threshold_dram_pmem(
            ranking,
            std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max()).threshVal;
    ASSERT_EQ(thresh_lowest, thresh_lowest_pmem); // double for equality
    ASSERT_EQ(thresh_lowest, 0);
    ASSERT_EQ(ranking_is_hot(ranking, &blocks[0]), false);
    for (size_t i=1; i<BLOCKS_SIZE; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), true);
    }
}

TEST_F(RankingTest, check_hotness_50_50) {
    double RATIO_EQUAL=0.5;
    // equal by size
    // total size allocated:
    // (1.+BLOCKS_SIZE)/2*BLOCKS_SIZE (arithmetic series)
    // half the size:
    // (1.+BLOCKS_SIZE)/4*BLOCKS_SIZE = sn
    double HALF_SIZE=(1.+BLOCKS_SIZE)/4*BLOCKS_SIZE;
    // how many previous elements have half the size:
    // (1.+n)/2*n=sn => 1+n=2*sn/n => n^2+n-2*sn=0 => delta=1+8*sn
    double delta = 1+8*HALF_SIZE;
    size_t n = floor((-1+sqrt(delta))/2);
    ASSERT_EQ(n, 70u); // calculated by hand
    double thresh_equal =
        ranking_calculate_hot_threshold_dram_total(
            ranking, RATIO_EQUAL, RATIO_EQUAL).threshVal;
    double thresh_equal_pmem =
        ranking_calculate_hot_threshold_dram_pmem(ranking, 1, 1).threshVal;
    ASSERT_EQ(thresh_equal, thresh_equal_pmem);
#if QUANTIFICATION_ENABLED
    double ACCURACY = 1e-9;
    ASSERT_EQ(thresh_equal, quantify_dequantify(29));
    ASSERT_LE(abs(20.085536923187668-quantify_dequantify(29)), ACCURACY);
    for (size_t i=0; i<21; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
    for (size_t i=21; i<100; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), true);
    }
    ASSERT_EQ(BLOCKS_SIZE, 100u);
#else
    ASSERT_EQ(thresh_equal, 29);
    for (size_t i=0; i<30; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
    for (size_t i=30; i<100; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), true);
    }
    ASSERT_EQ(BLOCKS_SIZE, 100u);
#endif
}

TEST_F(RankingTest, check_hotness_50_50_removed) {
    const size_t SUBSIZE=10u;
    for (size_t i=SUBSIZE; i<BLOCKS_SIZE; ++i) {
        ranking_remove(ranking, blocks[i].f, blocks[i].num_allocs);
    }
    double RATIO_EQUAL_TOTAL=0.5;
    double RATIO_EQUAL_PMEM=1;
    double thresh_equal = ranking_calculate_hot_threshold_dram_total(
        ranking, RATIO_EQUAL_TOTAL, RATIO_EQUAL_TOTAL).threshVal;
    double thresh_equal_pmem = ranking_calculate_hot_threshold_dram_pmem(
        ranking, RATIO_EQUAL_PMEM, RATIO_EQUAL_PMEM).threshVal;
    // hand calculations:
    // 100, 99, 98, 97, 96, 95, 94, 93, 92, 91
    // sum:
    // 100, 199, 297, 394, 490 <- this is the one we are looking for
#if QUANTIFICATION_ENABLED
    double ACCURACY=1e-9;
    ASSERT_EQ(thresh_equal, quantify_dequantify(4));
    ASSERT_LE(abs(thresh_equal-quantify_dequantify(4)), ACCURACY);
    ASSERT_EQ(thresh_equal, thresh_equal_pmem);
    for (size_t i=0; i<3; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
    for (size_t i=3; i<10; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), true);
    }
    ASSERT_EQ(BLOCKS_SIZE, 100u);
    ASSERT_EQ(SUBSIZE, 10u);
#else
    ASSERT_EQ(thresh_equal, 4);
    ASSERT_EQ(thresh_equal, thresh_equal_pmem);
    for (size_t i=0; i<5; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
    for (size_t i=5; i<10; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), true);
    }
    ASSERT_EQ(BLOCKS_SIZE, 100u);
    ASSERT_EQ(SUBSIZE, 10u);
#endif
}

class RankingTestSameHotness: public ::testing::Test
{
protected:
    ranking_t *ranking;
    static constexpr size_t BLOCKS_SIZE=100u;
    struct ttype blocks[BLOCKS_SIZE];
private:
    void SetUp()
    {
        ranking_create(&ranking, 0.9);

        for (size_t i=0; i<BLOCKS_SIZE; ++i) {
            blocks[i].num_allocs=BLOCKS_SIZE-i;
            blocks[i].f=i%50;
            ranking_add(ranking, blocks[i].f, blocks[i].num_allocs);
        }

    }
    void TearDown()
    {
        ranking_destroy(ranking);
    }
};
constexpr size_t RankingTestSameHotness::BLOCKS_SIZE;

// initialized
TEST_F(RankingTestSameHotness, check_hotness_highest) {
    double RATIO_PMEM_ONLY_TOTAL=0;
    double RATIO_PMEM_ONLY_PMEM=0;
    double thresh_highest = ranking_calculate_hot_threshold_dram_total(
        ranking, RATIO_PMEM_ONLY_TOTAL, RATIO_PMEM_ONLY_TOTAL).threshVal;
    double thresh_highest_pmem = ranking_calculate_hot_threshold_dram_pmem(
        ranking, RATIO_PMEM_ONLY_PMEM, RATIO_PMEM_ONLY_PMEM).threshVal;
#if QUANTIFICATION_ENABLED
    double ACCURACY=1e-9;
    ASSERT_EQ(thresh_highest, quantify_dequantify((BLOCKS_SIZE-1)%50));
    ASSERT_EQ(thresh_highest, quantify_dequantify(49));
    ASSERT_LE(abs(quantify_dequantify(49)-20.085536923187668), ACCURACY);
    ASSERT_EQ(thresh_highest, thresh_highest_pmem);
    for (volatile size_t i=0; i<21; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
    for (volatile size_t i=21; i<50; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), true);
    }
    for (volatile size_t i=50; i<71; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
    for (volatile size_t i=71; i<BLOCKS_SIZE; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), true);
    }
#else
    ASSERT_EQ(thresh_highest, (BLOCKS_SIZE-1)%50);
    ASSERT_EQ(thresh_highest, 49);
    ASSERT_EQ(thresh_highest, thresh_highest_pmem);
    for (size_t i=0; i<BLOCKS_SIZE; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
#endif
//     ASSERT_EQ(ranking_is_hot(ranking, &blocks[BLOCKS_SIZE-1]), true);
}

TEST_F(RankingTestSameHotness, check_hotness_lowest) {
    double RATIO_DRAM_ONLY=1;
    double thresh_lowest = ranking_calculate_hot_threshold_dram_total(
        ranking, RATIO_DRAM_ONLY, RATIO_DRAM_ONLY).threshVal;
    double thresh_lowest_pmem = ranking_calculate_hot_threshold_dram_pmem(
        ranking,
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max()).threshVal;
    ASSERT_EQ(thresh_lowest, 0);
    ASSERT_EQ(thresh_lowest, thresh_lowest_pmem);
    for (size_t i=0; i<BLOCKS_SIZE; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), i%50 != 0);
    }
}

TEST_F(RankingTestSameHotness, check_hotness_50_50) {
    double RATIO_EQUAL_TOTAL=0.5;
    double RATIO_EQUAL_PMEM=1;
    // when grouped in pairs, we get 150, 148, .., 52
    // arithmetic series a0 = 150, r=-2, n=50
    // we now want to find n_50: s_n_50 = s_n/2
    // sn == 5050 => sn/2 == 2525
    // the equation to solve for n_50:
    // 2525 == n_50*(150-2*(n_50-1))/2
    // 5050 = 150*n_50-2*n_50^2+2*n_50 = 152*n_50-2*n_50^2
    // n_50^2-76*n_50+2525 = 0
    // delta = 76^2-4*2525 = 5776-10000
    double thresh_equal = ranking_calculate_hot_threshold_dram_total(
        ranking, RATIO_EQUAL_TOTAL, RATIO_EQUAL_TOTAL).threshVal;
    double thresh_equal_pmem = ranking_calculate_hot_threshold_dram_pmem(
        ranking, RATIO_EQUAL_PMEM, RATIO_EQUAL_PMEM).threshVal;
#if QUANTIFICATION_ENABLED
    double ACCURACY=1e-9;
    ASSERT_EQ(thresh_equal, quantify_dequantify(19));
    ASSERT_LE(abs(quantify_dequantify(19)-7.38905609893065), ACCURACY);
    ASSERT_EQ(thresh_equal, thresh_equal_pmem);
    for (volatile size_t i=0; i<8; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
    for (volatile size_t i=8; i<50; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), true);
    }
    for (volatile size_t i=50; i<58; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
    for (volatile size_t i=58; i<100; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), true);
    }
    ASSERT_EQ(BLOCKS_SIZE, 100u);
#else
    ASSERT_EQ(thresh_equal, 19);
    ASSERT_EQ(thresh_equal, thresh_equal_pmem);
    for (size_t i=0; i<20; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
    for (size_t i=20; i<50; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), true);
    }
    for (size_t i=50; i<70; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
    for (size_t i=70; i<100; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), true);
    }
    ASSERT_EQ(BLOCKS_SIZE, 100u);
#endif
}

TEST_F(RankingTestSameHotness, check_hotness_50_50_removed) {
    const size_t SUBSIZE=10u;
    for (size_t i=SUBSIZE; i<BLOCKS_SIZE; ++i) {
        ranking_remove(ranking, blocks[i].f, blocks[i].num_allocs);
    }
    double RATIO_EQUAL_TOTAL=0.5;
    double RATIO_EQUAL_PMEM=1;
    double thresh_equal = ranking_calculate_hot_threshold_dram_total(
        ranking, RATIO_EQUAL_TOTAL, RATIO_EQUAL_TOTAL).threshVal;
    double thresh_equal_pmem = ranking_calculate_hot_threshold_dram_pmem(
        ranking, RATIO_EQUAL_PMEM, RATIO_EQUAL_PMEM).threshVal;
    // hand calculations:
    // 100, 99, 98, 97, 96, 95, 94, 93, 92, 91
    // sum:
    // 100, 199, 297, 394, 490 <- this is the one we are looking for
#if QUANTIFICATION_ENABLED
    double ACCURACY=1e-9;
    ASSERT_EQ(thresh_equal, quantify_dequantify(4));
    ASSERT_LE(abs(quantify_dequantify(4)-2.718281828459045), ACCURACY);
    ASSERT_EQ(thresh_equal, thresh_equal_pmem);
    for (size_t i=0; i<3; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
    for (size_t i=3; i<10; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), true);
    }
    ASSERT_EQ(BLOCKS_SIZE, 100u);
    ASSERT_EQ(SUBSIZE, 10u);
#else
    ASSERT_EQ(thresh_equal, 4);
    ASSERT_EQ(thresh_equal, thresh_equal_pmem);
    for (size_t i=0; i<5; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
    for (size_t i=5; i<10; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), true);
    }
    ASSERT_EQ(BLOCKS_SIZE, 100u);
    ASSERT_EQ(SUBSIZE, 10u);
#endif
}

extern "C" {
typedef struct {
    uint32_t val;
    size_t weight;
} wre_test_struct_t;

static bool is_lower_int(const void* i1, const void* i2) {
    return ((wre_test_struct_t*)i1)->val < ((wre_test_struct_t*)i2)->val;
}
}

class WreTreeTest: public ::testing::Test
{
protected:
    static constexpr size_t TAB_SIZE=100u;
    static constexpr size_t EXTENDED_TAB_SIZE=200u;
    wre_test_struct_t blocks[EXTENDED_TAB_SIZE];
    wre_tree_t *tree;
private:
    void SetUp()
    {
        for (size_t i=0; i<EXTENDED_TAB_SIZE; ++i) {
            blocks[i].val=i;
            blocks[i].weight=abs(((int64_t)TAB_SIZE)-(int64_t)i);
        }
        wre_create(&tree, is_lower_int);
    }
    void TearDown()
    {
        wre_destroy(tree);
    }
};

TEST_F(WreTreeTest, simple_adds) {
    wre_put(tree, &blocks[6], blocks[6].weight); // value 6, weight: 94
    ASSERT_EQ(tree->rootNode->subtreeWeight, 94u);
    ASSERT_EQ(tree->rootNode->height, 0u);
    ASSERT_EQ(tree->rootNode->left, nullptr);
    ASSERT_EQ(tree->rootNode->right, nullptr);
    ASSERT_EQ(tree->rootNode->parent, nullptr);
    ASSERT_EQ(tree->rootNode->which, ROOT_NODE);
    ASSERT_EQ(tree->rootNode->data, &blocks[6]);
    ASSERT_EQ(((wre_test_struct_t*)tree->rootNode->data)->val, 6u);
    ASSERT_EQ(((wre_test_struct_t*)tree->rootNode->data)->weight, 94u);
    wre_put(tree, &blocks[3], blocks[3].weight); // value 3, weight: 97
    ASSERT_EQ(tree->rootNode->subtreeWeight, 191u);
    ASSERT_EQ(tree->rootNode->height, 1u);
    ASSERT_EQ(tree->rootNode->left->subtreeWeight, 97u);
    ASSERT_EQ(tree->rootNode->left->height, 0u);
    ASSERT_EQ(tree->rootNode->left->left, nullptr);
    ASSERT_EQ(tree->rootNode->left->right, nullptr);
    ASSERT_EQ(tree->rootNode->left->parent, tree->rootNode);
    ASSERT_EQ(tree->rootNode->left->which, LEFT_NODE);
    ASSERT_EQ(tree->rootNode->right, nullptr);
    ASSERT_EQ(tree->rootNode->parent, nullptr);
    ASSERT_EQ(tree->rootNode->which, ROOT_NODE);
    ASSERT_EQ(tree->rootNode->data, &blocks[6]);
    ASSERT_EQ(((wre_test_struct_t*)tree->rootNode->data)->val, 6u);
    ASSERT_EQ(((wre_test_struct_t*)tree->rootNode->data)->weight, 94u);
}

TEST_F(WreTreeTest, simple_adds_removes) {
    wre_put(tree, &blocks[6], blocks[6].weight); // value 6, weight: 94
    ASSERT_EQ(tree->rootNode->subtreeWeight, 94u);
    ASSERT_EQ(tree->rootNode->height, 0u);
    ASSERT_EQ(tree->rootNode->left, nullptr);
    ASSERT_EQ(tree->rootNode->right, nullptr);
    ASSERT_EQ(tree->rootNode->parent, nullptr);
    ASSERT_EQ(tree->rootNode->which, ROOT_NODE);
    ASSERT_EQ(tree->rootNode->data, &blocks[6]);
    ASSERT_EQ(((wre_test_struct_t*)tree->rootNode->data)->val, 6u);
    ASSERT_EQ(((wre_test_struct_t*)tree->rootNode->data)->weight, 94u);
    void* data = wre_remove(tree, &blocks[6]);
    ASSERT_EQ(data, &blocks[6]);
    ASSERT_EQ(tree->rootNode, nullptr);
    wre_put(tree, &blocks[6], blocks[6].weight); // value 6, weight: 94
    wre_put(tree, &blocks[3], blocks[3].weight); // value 3, weight: 97
    ASSERT_EQ(tree->rootNode->subtreeWeight, 191u);
    ASSERT_EQ(tree->rootNode->height, 1u);
    ASSERT_EQ(tree->rootNode->left->subtreeWeight, 97u);
    ASSERT_EQ(tree->rootNode->left->height, 0u);
    ASSERT_EQ(tree->rootNode->left->left, nullptr);
    ASSERT_EQ(tree->rootNode->left->right, nullptr);
    ASSERT_EQ(tree->rootNode->left->parent, tree->rootNode);
    ASSERT_EQ(tree->rootNode->left->which, LEFT_NODE);
    ASSERT_EQ(tree->rootNode->right, nullptr);
    ASSERT_EQ(tree->rootNode->parent, nullptr);
    ASSERT_EQ(tree->rootNode->which, ROOT_NODE);
    ASSERT_EQ(tree->rootNode->data, &blocks[6]);
    ASSERT_EQ(((wre_test_struct_t*)tree->rootNode->data)->val, 6u);
    ASSERT_EQ(((wre_test_struct_t*)tree->rootNode->data)->weight, 94u);
    data = wre_remove(tree, &blocks[3]);
    ASSERT_EQ(data, &blocks[3]);
    ASSERT_EQ(tree->rootNode->subtreeWeight, 94u);
    ASSERT_EQ(tree->rootNode->height, 0u);
    ASSERT_EQ(tree->rootNode->left, nullptr);
    ASSERT_EQ(tree->rootNode->right, nullptr);
    ASSERT_EQ(tree->rootNode->parent, nullptr);
    ASSERT_EQ(tree->rootNode->which, ROOT_NODE);
    ASSERT_EQ(tree->rootNode->data, &blocks[6]);
    ASSERT_EQ(((wre_test_struct_t*)tree->rootNode->data)->val, 6u);
    ASSERT_EQ(((wre_test_struct_t*)tree->rootNode->data)->weight, 94u);
    wre_put(tree, &blocks[3], blocks[3].weight); // value 3, weight: 97
    data = wre_remove(tree, &blocks[6]);
    ASSERT_EQ(data, &blocks[6]);
    ASSERT_EQ(tree->rootNode->subtreeWeight, 97u);
    ASSERT_EQ(tree->rootNode->height, 0u);
    ASSERT_EQ(tree->rootNode->left, nullptr);
    ASSERT_EQ(tree->rootNode->right, nullptr);
    ASSERT_EQ(tree->rootNode->parent, nullptr);
    ASSERT_EQ(tree->rootNode->which, ROOT_NODE);
    ASSERT_EQ(tree->rootNode->data, &blocks[3]);
    ASSERT_EQ(((wre_test_struct_t*)tree->rootNode->data)->val, 3u);
    ASSERT_EQ(((wre_test_struct_t*)tree->rootNode->data)->weight, 97u);
}

TEST_F(WreTreeTest, add_multiple_nodes) {
    size_t accumulated_weight=0u;
    // add all nodes in regular order
    for (size_t i=0; i<TAB_SIZE; ++i) {
        wre_put(tree, &blocks[i], blocks[i].weight);
        accumulated_weight += blocks[i].weight;
        ASSERT_EQ(tree->rootNode->subtreeWeight, accumulated_weight);
    }
    ASSERT_EQ(tree->rootNode->height, 6u);
    // check contents
    ASSERT_EQ(tree->rootNode->subtreeWeight, 5050u);
    // TODO more checks!
}

TEST_F(WreTreeTest, add_remove_multiple_nodes) {
    size_t accumulated_weight=0u;
    // add all nodes in regular order
    for (size_t i=0; i<EXTENDED_TAB_SIZE; ++i) {
        wre_put(tree, &blocks[i], blocks[i].weight);
        accumulated_weight += blocks[i].weight;
        ASSERT_EQ(tree->rootNode->subtreeWeight, accumulated_weight);
    }
    ASSERT_EQ(tree->rootNode->height, 7u);
    for (size_t i=TAB_SIZE; i<EXTENDED_TAB_SIZE; ++i) {
        void* removed = wre_remove(tree, &blocks[i]);
        ASSERT_EQ(removed, &blocks[i]);
        accumulated_weight -= blocks[i].weight;
        ASSERT_EQ(tree->rootNode->subtreeWeight, accumulated_weight);
    }
    ASSERT_EQ(tree->rootNode->height, 6u);
    // check contents
    ASSERT_EQ(tree->rootNode->subtreeWeight, 5050u);
}

TEST_F(WreTreeTest, add_remove_multiple_modes_desc) {
    size_t accumulated_weight=0u;
    // add all nodes in regular order
    for (size_t i=0; i<EXTENDED_TAB_SIZE; ++i) {
        wre_put(tree, &blocks[i], blocks[i].weight);
        accumulated_weight += blocks[i].weight;
        ASSERT_EQ(tree->rootNode->subtreeWeight, accumulated_weight);
    }
    ASSERT_EQ(tree->rootNode->height, 7u);
    for (size_t i=EXTENDED_TAB_SIZE-1; i>=TAB_SIZE; --i) {
        void* removed = wre_remove(tree, &blocks[i]);
        ASSERT_EQ(removed, &blocks[i]);
        accumulated_weight -= blocks[i].weight;
        ASSERT_EQ(tree->rootNode->subtreeWeight, accumulated_weight);
    }
    ASSERT_EQ(tree->rootNode->height, 6u);
    // check contents
    ASSERT_EQ(tree->rootNode->subtreeWeight, 5050u);
}

// ----------------- hotness integration tests

typedef struct TouchCbArg {
    char *name;
    size_t counter;
} TouchCbArg_t;

void touch_cb(void *arg) {
    TouchCbArg_t *cb_arg = (TouchCbArg_t*) arg;
//     printf("touch [%s]\n", cb_arg->name);
    cb_arg->counter++;
}

static std::vector<TouchCbArg_t*> g_cbArgs; // for debugging purposes

class TestBuffer {
//     const size_t BUFF_SIZE = 1e9; // 1 GB
    const double frequency; /// @pre  0 < frequency <= 1
    double accumulated_freq=0;
    RandomIncremeneter incrementer;

public:
    volatile uint8_t *data;
    const size_t BUFF_SIZE = 2e8; // 200 MB

    void operator =(const TestBuffer &mat)=delete;
    // required for vector<TestMatrix>::reserve:
    // TestMatrix(const TestMatrix &mat)=delete;
    TestBuffer(double freq) : frequency(freq) {
        if (frequency > 1 || frequency <=0) {
            std::string error_info("Incorrect frequency: ");
            error_info += frequency;
            throw std::runtime_error(error_info);
        }

        data = nullptr;
    }

    virtual ~TestBuffer() {
        // DATA IS NOT FREED!
        // would be best to have a reference counter, or sth - shared ptr?
    }

    void DoSomeWork() {
        accumulated_freq+=frequency;
        if (accumulated_freq>=1) {
            incrementer.IncrementRandom(data, BUFF_SIZE);
            accumulated_freq -=1.;
        }
    }

    uint64_t CalculateSum() {
        uint64_t sum=0u;
        if (data) {
            for (size_t i=0u; i<BUFF_SIZE; ++i) {
                sum += data[i];
            }
        }
        return sum;
    }

    memkind_t DetectKind() {
        return memkind_detect_kind((void*)data);
    }

    double GetHotness() {
        return tachanka_get_addr_hotness((void*)data);
    }

    Hotness_e GetHotnessType() {
        return tachanka_get_hotness_type((void*)data);
    }

protected:

    TouchCbArg_t *AllocCreateCbArg() {
        static size_t counter=0;
        std::string name = std::string("buff_")+std::to_string(counter);
        printf("AllocCreateCbArg: allocated data [%s] at %p, size: [%lu]\n", name.c_str(), data, BUFF_SIZE);
        size_t str_size = strlen(name.c_str()) +1u;
        // use regular mallocs here - should not be an issue
        TouchCbArg_t *cb_arg = (TouchCbArg_t*)malloc(sizeof(TouchCbArg_t));
        cb_arg->name = (char*)malloc(sizeof(char)*str_size);
        cb_arg->counter = 0u;
        snprintf(cb_arg->name, str_size, "%s", name.c_str());
        ++counter;
        return cb_arg;
    }

    static void FreeCbArg(TouchCbArg_t *arg) {
        free(arg->name);
        free(arg);
    }

    void RegisterCallback() {
        TouchCbArg_t *cb_arg = AllocCreateCbArg();
        // TODO create event - set touch callback and add it to the queue
//         int ret = tachanka_set_touch_callback(data, touch_cb, cb_arg);
        EventEntry_t entry;
        entry.type = EVENT_SET_TOUCH_CALLBACK,
        entry.data.touchCallbackData.address = (void*)data;
        entry.data.touchCallbackData.callback = touch_cb;
        entry.data.touchCallbackData.callbackArg = cb_arg;
        bool ret = tachanka_ranking_event_push(&entry);
        ASSERT_TRUE(ret);
        g_cbArgs.push_back(cb_arg);
    }

    virtual void FreeData() {
        memtier_free((void*)data);
        data=nullptr;
    }

    virtual void AllocData(struct memtier_memory *m) {
        memtier_free((void*)data);
        data = (uint8_t*)memtier_malloc(m, BUFF_SIZE);
        RegisterCallback();
    }

    virtual void ReallocData(struct memtier_memory *m) {
        uint8_t *data_temp = (uint8_t*)memtier_malloc(m, BUFF_SIZE);
        memtier_free((void*)data);
        data = data_temp;
        RegisterCallback();
    }
};

// MEMKIND_NO_INLINE is not necessary, these functions are virtual
// stays here for human visibility - the intent is more verbose and explicit
#define MEMKIND_NO_INLINE __attribute__((noinline))

// define TestMatrixA and TestMatrixB
// both of them support the same operations
// but their Free/Alloc have different backtrace

class TestBufferA : public TestBuffer {
public:
    MEMKIND_NO_INLINE TestBufferA(struct memtier_memory *m, double freq) :
        TestBuffer(freq) {
        AllocData(m);
    }
    MEMKIND_NO_INLINE void FreeData() {
        return TestBuffer::FreeData();
    }
    MEMKIND_NO_INLINE void AllocData(struct memtier_memory *m) {
        std::mutex mut;
        // mutex: avoid optimisations
        std::unique_lock<std::mutex> lock(mut);
        volatile char buff[100];
        for (size_t i=0; i< sizeof(buff)/sizeof(buff[0]); ++i)
            buff[i]=0xFF;
        lock.unlock();
        return TestBuffer::AllocData(m);
    }
    MEMKIND_NO_INLINE void ReallocData(struct memtier_memory *m) {
        std::mutex mut;
        // mutex: avoid optimisations
        std::unique_lock<std::mutex> lock(mut);
        volatile char buff[100];
        for (size_t i=0; i< sizeof(buff)/sizeof(buff[0]); ++i)
            buff[i]=0xFF;
        lock.unlock();
        return TestBuffer::ReallocData(m);
    }
};

class TestBufferB : public TestBuffer {
public:
    MEMKIND_NO_INLINE TestBufferB(struct memtier_memory *m, double freq) :
        TestBuffer(freq) {
        AllocData(m);
    }

    MEMKIND_NO_INLINE void FreeData() {
        return TestBuffer::FreeData();
    }
    MEMKIND_NO_INLINE void AllocData(struct memtier_memory *m) {
        std::mutex mut;
        // mutex: avoid optimisations
        std::unique_lock<std::mutex> lock(mut);
        volatile char buff[200];
        for (size_t i=0; i< sizeof(buff)/sizeof(buff[0]); ++i)
            buff[i]=0xFF;
        lock.unlock();
        return TestBuffer::AllocData(m);
    }
    MEMKIND_NO_INLINE void ReallocData(struct memtier_memory *m) {
        std::mutex mut;
        // mutex: avoid optimisations
        std::unique_lock<std::mutex> lock(mut);
        volatile char buff[200];
        for (size_t i=0; i< sizeof(buff)/sizeof(buff[0]); ++i)
            buff[i]=0xFF;
        lock.unlock();
        return TestBuffer::ReallocData(m);
    }
};

class TestBufferC : public TestBuffer {
public:
    MEMKIND_NO_INLINE TestBufferC(struct memtier_memory *m, double freq) :
        TestBuffer(freq) {
        AllocData(m);
    }

    MEMKIND_NO_INLINE void FreeData() {
        return TestBuffer::FreeData();
    }
    MEMKIND_NO_INLINE void AllocData(struct memtier_memory *m) {
        std::mutex mut;
        // mutex: avoid optimisations
        std::unique_lock<std::mutex> lock(mut);
        volatile char buff[300];
        for (size_t i=0; i< sizeof(buff)/sizeof(buff[0]); ++i)
            buff[i]=0xFF;
        lock.unlock();
        return TestBuffer::AllocData(m);
    }
    MEMKIND_NO_INLINE void ReallocData(struct memtier_memory *m) {
        std::mutex mut;
        // mutex: avoid optimisations
        std::unique_lock<std::mutex> lock(mut);
        volatile char buff[300];
        for (size_t i=0; i< sizeof(buff)/sizeof(buff[0]); ++i)
            buff[i]=0xFF;
        lock.unlock();
        return TestBuffer::ReallocData(m);
    }
};

class IntegrationHotnessSingleTest: public ::testing::Test
{
protected:
    // base frequency of 1, 1/2, 1/3, 1/4, ...
    std::vector<TestBufferA> bufferA;
    // base frequency of 1/2, 1/3, 1/4, 1/5, ...
    std::vector<TestBufferB> bufferB;
    std::vector<TestBufferC> bufferC;
//     const size_t ALLOCATED_SIZE=5e8; // 500 MB
    const size_t ALLOCATED_SIZE=1e9; // 1 GB

    static constexpr size_t MATRICES_SIZE=1u;
    struct memtier_memory *m_tier_memory;
private:
    void SetUp()
    {
        struct memtier_builder *m_builder =
            memtier_builder_new(MEMTIER_POLICY_DATA_HOTNESS);

        int res = memtier_builder_add_tier(m_builder, MEMKIND_DEFAULT, 1); // add dram kind
        ASSERT_EQ(0, res);
        res = memtier_builder_add_tier(m_builder, MEMKIND_REGULAR, 1); // add pmem kind
        ASSERT_EQ(0, res);
        m_tier_memory = memtier_builder_construct_memtier_memory(m_builder);
        ASSERT_NE(nullptr, m_tier_memory);

        bufferA.reserve(MATRICES_SIZE);
        bufferB.reserve(MATRICES_SIZE);
        bufferC.reserve(MATRICES_SIZE);
        for (size_t i=0; i<MATRICES_SIZE; ++i) {
            double base_frequency = 1./(i+1);
            bufferA.push_back(TestBufferA(m_tier_memory, base_frequency));
            bufferB.push_back(TestBufferB(m_tier_memory, base_frequency/2));
            bufferC.push_back(TestBufferC(m_tier_memory, (3*base_frequency/4)));
        }
        memtier_builder_delete(m_builder);
    }

    void TearDown()
    {
        for (TouchCbArg_t *arg : g_cbArgs) {
            free(arg->name);
            free(arg);
        }
        g_cbArgs.clear();
        memtier_delete_memtier_memory(m_tier_memory);
    }
};

class IntegrationHotnessMultipleTest: public ::testing::Test
{
protected:
    // base frequency of 1, 1/2, 1/3, 1/4, ...
    std::vector<TestBufferA> bufferA;
    // base frequency of 1/2, 1/3, 1/4, 1/5, ...
    std::vector<TestBufferB> bufferB;
    const size_t ALLOCATED_SIZE=5e8; // 500 MB
    std::vector<TestBufferC> bufferC;

    static constexpr size_t MATRICES_SIZE=1u;
    struct memtier_memory *m_tier_memory;
private:
    void SetUp()
    {
        struct memtier_builder *m_builder =
            memtier_builder_new(MEMTIER_POLICY_DATA_HOTNESS);

        int res = memtier_builder_add_tier(m_builder, MEMKIND_DEFAULT, 1);
        ASSERT_EQ(0, res);
        m_tier_memory = memtier_builder_construct_memtier_memory(m_builder);
        ASSERT_NE(nullptr, m_tier_memory);

        bufferA.reserve(MATRICES_SIZE);
        bufferB.reserve(MATRICES_SIZE);
        bufferC.reserve(MATRICES_SIZE);
        for (size_t i=0; i<MATRICES_SIZE; ++i) {
            double base_frequency = 1./(i+1);
            bufferA.push_back(TestBufferA(m_tier_memory, base_frequency));
            bufferB.push_back(TestBufferB(m_tier_memory, base_frequency/2));
            bufferC.push_back(TestBufferC(m_tier_memory, (3*base_frequency/4)));
        }
        memtier_builder_delete(m_builder);
    }

    void TearDown()
    {
        memtier_delete_memtier_memory(m_tier_memory);
    }
};

// How the check for hotness should look like:
//  1) sort all objects/types by their frequency
//  2) calculate sum of all sizes
//  3) hand-calculate which objects should be cold and which ones hot
//  4) make a check in two loops: first loop for hot, second loop for cold
//
// Possible issues:
//  1) pebs thread has not done its work; mitigation:
//      a) wait (race condition-based mitigation)
//      b) explicitly call pebs thread
//  2) general race condition with pebs:
//      i) cases:
//          a) not enough: see point 1,
//          b) too many times: old time window is gone,
//          latest has 0 measurements
//      ii) mitigation:
//          a) explicitly call pebs, without separate thread
//
// For now, only "quickfix": make a test that's vulnerable to race condition


// next test: run

TEST_F(IntegrationHotnessSingleTest, test_random_hotness)
{
    // SIMPLE TEST - use only one Matrix per type
    TestBufferB &mb = bufferB[0]; // should do half the work of ma
    TestBufferC &mc = bufferC[0]; // should do (ma_work+mb_work)/2
    TestBufferA &ma = bufferA[0];
//     TestBufferC &mc = bufferC[0]; // should do (ma_work+mb_work)/2

    //printf("\nAddr range of bufferA: %p - %p\n", (char*)ma.data, (char*)(ma.data) + ma.BUFF_SIZE);
    //printf("Addr range of bufferB: %p - %p\n", (char*)mb.data, (char*)(mb.data) + mb.BUFF_SIZE);

    auto start_point = std::chrono::steady_clock::now();
    auto end_point = start_point;
    double millis_elapsed=0;
    const double LIMIT_MILLIS=15000;
    size_t iterations=0u;
    for (iterations=0; millis_elapsed < LIMIT_MILLIS; ++iterations) {
        ma.DoSomeWork();
        mb.DoSomeWork();
        mc.DoSomeWork();
        end_point = std::chrono::steady_clock::now();
        std::chrono::duration<double> duration = end_point-start_point;
        millis_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                .count();
    }
    double hotness_a=ma.GetHotness();
    double hotness_b=mb.GetHotness();
    double hotness_c=mc.GetHotness();
    size_t touches_a = g_cbArgs[0]->counter;
    size_t touches_b = g_cbArgs[1]->counter;
    size_t touches_c = g_cbArgs[2]->counter;

    double touch_ratio = ((double)touches_a)/touches_b;

    uint64_t asum = ma.CalculateSum();
    uint64_t bsum = mb.CalculateSum();
    uint64_t csum = mc.CalculateSum();
    // use calcualted data - prevent all loops from being optimized out
    printf("Total sums: A [%lu], B [%lu], C [%lu]\n", asum, bsum, csum);
    printf("Total touches: A [%lu], B [%lu], C [%lu]\n", touches_a, touches_b, touches_c);

    double ACCURACY = 0.6; // a little bit high... (bad, but seems code is ok)
    // check if sum ratio is as expected - a measure of work done
    double EXPECTED_RATIO = 2;
    double calculated_sum_ratio = ((double)asum)/bsum;
    ASSERT_LE(abs(touch_ratio - EXPECTED_RATIO), ACCURACY);
    ASSERT_LE(abs(calculated_sum_ratio - EXPECTED_RATIO), ACCURACY);

    const size_t MIN_SIGNIFICANT_WORK=(size_t)1e4;
    ASSERT_GT(iterations, MIN_SIGNIFICANT_WORK);

    // check if address is known and hotness was calculated
    ASSERT_GT(hotness_a, 0);
    ASSERT_GT(hotness_b, 0);
    ASSERT_GT(hotness_c, 0);

    // rough check
    ASSERT_GT(hotness_a, hotness_b);
    ASSERT_GT(hotness_a, hotness_c);
    ASSERT_GT(hotness_c, hotness_b);

    // check if hotness ratio is as expected
    double EXPECTED_HOTNESS_RATIO = 2;
    double calculated_hotness_ratio = hotness_a/hotness_b;
    ASSERT_LE(abs(calculated_hotness_ratio - EXPECTED_HOTNESS_RATIO), ACCURACY);

    Hotness_e a_type=ma.GetHotnessType();
    Hotness_e b_type=mb.GetHotnessType();
    Hotness_e c_type=mc.GetHotnessType();

    ASSERT_EQ(a_type, HOTNESS_HOT);
    ASSERT_EQ(b_type, HOTNESS_COLD);
    ASSERT_EQ(c_type, HOTNESS_NOT_FOUND); // when exactly equal thresh

    memkind_t a_kind = ma.DetectKind();
    memkind_t b_kind = mb.DetectKind();
    memkind_t c_kind = mb.DetectKind();

    // first, unknown allocations
    // check that all allocations were performed on DRAM
    ASSERT_EQ(a_kind, MEMKIND_DEFAULT);
    ASSERT_EQ(b_kind, MEMKIND_DEFAULT);
    ASSERT_EQ(c_kind, MEMKIND_DEFAULT);
    // TODO upgrade the test - this is not a good example, fallback to static
    // is same as detected kinds
}

TEST_F(IntegrationHotnessSingleTest, test_random_allocation_type)
{
    // this test has a very weird structure, it consists of:
    //      - a loop with two iterations,
    //      - internal state machine that checks iteration index,
    // this weird architecture is used to have exactly the same
    // backtrace per each realloc

    // SIMPLE TEST - use only one Matrix per type
    TestBufferB &mb = bufferB[0]; // should do half the work of ma
    TestBufferC &mc = bufferC[0]; // should do (ma_work+mb_work)/2
    TestBufferA &ma = bufferA[0];
//     TestBufferD &md = bufferD[0];

    for (volatile int iteration=0; iteration<2; ++iteration) {
        // reallocate data - constructor has different backtrace from Realloc
        ma.ReallocData(m_tier_memory);
        mb.ReallocData(m_tier_memory);
        mc.ReallocData(m_tier_memory);

        switch (iteration) {
            case 0: {

                auto start_point = std::chrono::steady_clock::now();
                auto end_point = start_point;
                double millis_elapsed=0;
                const double LIMIT_MILLIS=15000;
                size_t iterations=0u;

                memkind_t a_kind = ma.DetectKind();
                memkind_t b_kind = mb.DetectKind();
                memkind_t c_kind = mc.DetectKind();

                // check that all are on DRAM: initial allocation
                ASSERT_EQ(a_kind, MEMKIND_DEFAULT);
                ASSERT_EQ(b_kind, MEMKIND_DEFAULT);
                ASSERT_EQ(c_kind, MEMKIND_DEFAULT);

                for (iterations=0; millis_elapsed < LIMIT_MILLIS; ++iterations) {
                    ma.DoSomeWork();
                    mb.DoSomeWork();
                    mc.DoSomeWork();
                    end_point = std::chrono::steady_clock::now();
                    std::chrono::duration<double> duration = end_point-start_point;
                    millis_elapsed =
                        std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                            .count();
                }
                double hotness_a=ma.GetHotness();
                double hotness_b=mb.GetHotness();
            //     double hotness_c=mc.GetHotness(); not checked (yet)

                ASSERT_EQ(g_cbArgs.size(), 6u);
                size_t touches_a = g_cbArgs[0]->counter;
                size_t touches_b = g_cbArgs[1]->counter;
                size_t touches_c = g_cbArgs[2]->counter;

                ASSERT_EQ(touches_a, 0u);
                ASSERT_EQ(touches_b, 0u);
                ASSERT_EQ(touches_c, 0u);

                uint64_t asum = ma.CalculateSum();
                uint64_t bsum = mb.CalculateSum();
                uint64_t csum = mc.CalculateSum();
                // use calcualted data - prevent all loops from being optimized out
                printf("Total sums: A [%lu], B [%lu], C[%lu]\n", asum, bsum, csum);
                printf("Total touches - alloc1: A [%lu], B [%lu], C[%lu]\n", touches_a, touches_b, touches_c);

                touches_a = g_cbArgs[3]->counter;
                touches_b = g_cbArgs[4]->counter;
                touches_c = g_cbArgs[5]->counter;

                ASSERT_GT(touches_a, 0u);
                ASSERT_GT(touches_b, 0u);
                ASSERT_GT(touches_c, 0u);

                double touch_ratio = ((double)touches_a)/touches_b;

                printf("Total touches - alloc2: A [%lu], B [%lu], C[%lu]\n", touches_a, touches_b, touches_c);

                double ACCURACY = 0.6; // a little bit high... (bad, but seems code is ok)
                // check if sum ratio is as expected - a measure of work done
                double EXPECTED_RATIO = 2;
                double calculated_sum_ratio = ((double)asum)/bsum;
                ASSERT_LE(abs(touch_ratio - EXPECTED_RATIO), ACCURACY);
                ASSERT_LE(abs(calculated_sum_ratio - EXPECTED_RATIO), ACCURACY);

                const size_t MIN_SIGNIFICANT_WORK=(size_t)1e4;
                ASSERT_GT(iterations, MIN_SIGNIFICANT_WORK);

                // check if address is known and hotness was calculated
                ASSERT_GT(hotness_a, 0);
                ASSERT_GT(hotness_b, 0);

                // rough check
                ASSERT_GT(hotness_a, hotness_b);

                // check if hotness ratio is as expected
                double EXPECTED_HOTNESS_RATIO = 2;
                double calculated_hotness_ratio = hotness_a/hotness_b;
                ASSERT_LE(abs(calculated_hotness_ratio - EXPECTED_HOTNESS_RATIO), ACCURACY);

                Hotness_e a_type=ma.GetHotnessType();
                Hotness_e b_type=mb.GetHotnessType();
                Hotness_e c_type=mc.GetHotnessType();

                ASSERT_EQ(a_type, HOTNESS_HOT);
                ASSERT_EQ(b_type, HOTNESS_COLD);
                // exactly at thresh
                ASSERT_EQ(c_type, HOTNESS_NOT_FOUND);
                break;
            }
            case 1: {
                // block is registered asynchronously,
                // so hotness information will be visible with a delay
                // this delay is normally not a problem:
                //      1) allocations are done using hash, not address,
                //      2) all required manipulations are done from single
                //          (PEBS) thread, data races should either
                //          not be a concern, or should be handled
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                Hotness_e a_type=ma.GetHotnessType();
                Hotness_e b_type=mb.GetHotnessType();
                Hotness_e c_type=mc.GetHotnessType();
                ASSERT_EQ(a_type, HOTNESS_HOT);
                ASSERT_EQ(b_type, HOTNESS_COLD);
                // exactly at thresh
                ASSERT_EQ(c_type, HOTNESS_NOT_FOUND);


                memkind_t a_kind = ma.DetectKind();
                memkind_t b_kind = mb.DetectKind();
                memkind_t c_kind = mb.DetectKind();
                ASSERT_EQ(a_kind, MEMKIND_DEFAULT); // DRAM
                ASSERT_EQ(b_kind, MEMKIND_REGULAR); // PMEM
                ASSERT_EQ(c_kind, MEMKIND_REGULAR); // PMEM
                break;
            }
        }
    }
}
// TODO cleanup, maybe a separate file? currently, these tests take > min to complete


#include "stdlib.h"
#include "string.h"
#include "memkind/internal/ranking_queue.h"
#include "jemalloc/jemalloc.h"

// command to compile tests:
// gcc lockless_srmw_queue.c ranking_queue.c ranking_queue_tests.c -march=native -pthread -O3
// TODO move to gtest


static void test_simple(void) {
    //
    lq_buffer_t *buff;
    ranking_event_create(&buff, 4);
    EventEntry_t entry;
    bool empty_poppable = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(!empty_poppable);

    entry.type = EVENT_CREATE_ADD;
    entry.data.createAddData.hash = 1u;
    bool added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(added);

    entry.type = EVENT_TOUCH;
    entry.data.touchData.address = (void*)2u;
    added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(added);

    entry.type = EVENT_TOUCH;
    entry.data.touchData.address = (void*)3u;
    added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(added);

    entry.type = EVENT_CREATE_ADD;
    entry.data.createAddData.hash = 4u;
    added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(added);

    entry.type = EVENT_TOUCH;
    entry.data.createAddData.hash = 5u;
    added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(!added && "queue full!");

    bool popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(popped);
    ASSERT_TRUE(entry.type == EVENT_CREATE_ADD);
    ASSERT_TRUE(entry.data.createAddData.hash == 1u);

    popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(popped);
    ASSERT_TRUE(entry.type == EVENT_TOUCH);
    ASSERT_TRUE(entry.data.touchData.address == (void*) 2u);

    popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(popped);
    ASSERT_TRUE(entry.type == EVENT_TOUCH);
    ASSERT_TRUE(entry.data.touchData.address == (void*) 3u);

    popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(popped);
    ASSERT_TRUE(entry.type == EVENT_CREATE_ADD);
    ASSERT_TRUE(entry.data.createAddData.hash == 4u);

    popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(!popped);

    // queue empty, refill

    entry.type = EVENT_TOUCH;
    entry.data.touchData.address = (void*)6u;
    added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(added);

    entry.type = EVENT_TOUCH;
    entry.data.touchData.address = (void*)7u;
    added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(added);

    entry.type = EVENT_CREATE_ADD;
    entry.data.touchData.address = (void*)8u;
    added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(added);

    entry.type = EVENT_CREATE_ADD;
    entry.data.createAddData.hash = 9u;
    added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(added);

    entry.type = EVENT_TOUCH;
    entry.data.createAddData.hash = 10u;
    added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(!added && "queue full!");

    popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(popped);
    ASSERT_TRUE(entry.type == EVENT_TOUCH);
    ASSERT_TRUE(entry.data.touchData.address == (void*) 6u);

    popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(popped);
    ASSERT_TRUE(entry.type == EVENT_TOUCH);
    ASSERT_TRUE(entry.data.touchData.address == (void*) 7u);

    popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(popped);
    ASSERT_TRUE(entry.type == EVENT_CREATE_ADD);
    ASSERT_TRUE(entry.data.createAddData.hash == 8u);

    popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(popped);
    ASSERT_TRUE(entry.type == EVENT_CREATE_ADD);
    ASSERT_TRUE(entry.data.createAddData.hash == 9u);

    popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(!popped);

    ranking_event_destroy(buff);
}

static void test_simple_refill(void) {
    //
    lq_buffer_t *buff;
    ranking_event_create(&buff, 4);
    EventEntry_t entry;
    bool empty_poppable = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(!empty_poppable);

    entry.type = EVENT_CREATE_ADD;
    entry.data.createAddData.hash = 1u;
    bool added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(added);
    // 1 on queue, 3 empty

    entry.type = EVENT_TOUCH;
    entry.data.touchData.address = (void*)2u;
    added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(added);
    // 2 on queue, 2 empty

    entry.type = EVENT_TOUCH;
    entry.data.touchData.address = (void*)3u;
    added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(added);
    // 3 on queue, 1 empty

    bool popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(popped);
    ASSERT_TRUE(entry.type == EVENT_CREATE_ADD);
    ASSERT_TRUE(entry.data.createAddData.hash == 1u);
    // 2 on queue, 2 empty

    popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(popped);
    ASSERT_TRUE(entry.type == EVENT_TOUCH);
    ASSERT_TRUE(entry.data.touchData.address == (void*) 2u);
    // 1 on queue, 3 empty

    entry.type = EVENT_CREATE_ADD;
    entry.data.createAddData.hash = 4u;
    added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(added);
    // 2 on queue, 2 empty

    entry.type = EVENT_TOUCH;
    entry.data.touchData.address = (void*)6u;
    added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(added);
    // 3 on queue, 1 empty

    entry.type = EVENT_TOUCH;
    entry.data.touchData.address = (void*)7u;
    added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(added);
    // 4 on queue, 0 empty

    // queue full
    entry.type = EVENT_CREATE_ADD;
    entry.data.touchData.address = (void*)8u;
    added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(!added);
    // 4 on queue, 0 empty

    popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(popped);
    ASSERT_TRUE(entry.type == EVENT_TOUCH);
    ASSERT_TRUE(entry.data.createAddData.hash == 3u);
    // 3 on queue, 1 empty

    popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(popped);
    ASSERT_TRUE(entry.type == EVENT_CREATE_ADD);
    ASSERT_TRUE(entry.data.createAddData.hash == 4u);
    // 2 on queue, 2 empty

    popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(popped);
    ASSERT_TRUE(entry.type == EVENT_TOUCH);
    ASSERT_TRUE(entry.data.touchData.address == (void*) 6u);
    // 1 on queue, 3 empty

    entry.type = EVENT_CREATE_ADD;
    entry.data.createAddData.hash = 9u;
    added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(added);
    // 2 on queue, 2 empty

    entry.type = EVENT_CREATE_ADD;
    entry.data.createAddData.hash = 10u;
    added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(added);
    // 3 on queue, 1 empty

    entry.type = EVENT_CREATE_ADD;
    entry.data.createAddData.hash = 11u;
    added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(added);
    // 4 on queue, 0 empty

    entry.type = EVENT_TOUCH;
    entry.data.createAddData.hash = 12u;
    added = ranking_event_push(buff, &entry);
    ASSERT_TRUE(!added && "queue full!");
    // 4 on queue, 0 empty

    popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(popped);
    ASSERT_TRUE(entry.type == EVENT_TOUCH);
    ASSERT_TRUE(entry.data.touchData.address == (void*) 7u);
    // 3 on queue, 1 empty

    popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(popped);
    ASSERT_TRUE(entry.type == EVENT_CREATE_ADD);
    ASSERT_TRUE(entry.data.createAddData.hash == 9u);
    // 2 on queue, 2 empty

    popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(popped);
    ASSERT_TRUE(entry.type == EVENT_CREATE_ADD);
    ASSERT_TRUE(entry.data.createAddData.hash == 10u);
    // 1 on queue, 3 empty

    popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(popped);
    ASSERT_TRUE(entry.type == EVENT_CREATE_ADD);
    ASSERT_TRUE(entry.data.createAddData.hash == 11u);
    // 0 on queue, 4 empty

    popped = ranking_event_pop(buff, &entry);
    ASSERT_TRUE(!popped);

    ranking_event_destroy(buff);
}

typedef struct TestDataWriter {
    EventEntry_t *entries;
    size_t entriesSize;
    lq_buffer_t *buff;
} TestDataWriter;

typedef struct TestDataReader {
    EventEntry_t *dest;
    size_t destSize;
    lq_buffer_t *buff;
} TestDataReader;

static void *write_batch(void* data) {
    TestDataWriter *data_ = (TestDataWriter*)data;
    // TODO
    for (size_t i=0; i<data_->entriesSize; ++i) {
        while (!ranking_event_push(data_->buff, &data_->entries[i]));
    }

    return NULL;
}

static void *read_batch(void* data) {
    TestDataReader *data_ = (TestDataReader*)data;
    for (size_t i=0; i<data_->destSize; ++i) {
        // TODO
        EventEntry_t temp;
        while (!ranking_event_pop(data_->buff, &temp));
//         ASSERT_TRUE(temp.type == EVENT_TOUCH);
//         ASSERT_TRUE(((size_t)temp.data.touchData.address) < data_->destSize);
        assert(temp.type == EVENT_TOUCH);
        assert(((size_t)temp.data.touchData.address) < data_->destSize);
        data_->dest[(size_t)temp.data.touchData.address] = temp;
    }
    return NULL;
}

#include "pthread.h"

static void stress_test_simple(size_t writers, size_t params_per_thread, size_t buffer_size, size_t iterations) {
    // scenario:
    //  - create source array,
    //  - distribute source array chunks between writers,
    //  - write and read simultaneously (all writers, one reader),
    //  - check that all elements were correctly read

    size_t source_size = writers*params_per_thread;
    EventEntry_t *entries_source = (EventEntry_t*)malloc(source_size*sizeof(EventEntry_t));
    EventEntry_t *entries_dest = (EventEntry_t*)calloc(source_size, sizeof(EventEntry_t));
    // TODO init entries source and dest
    for (size_t i=0; i<source_size; ++i) {
        entries_source[i].type=EVENT_TOUCH;
        entries_source[i].data.touchData.address=(void*)i;
    }
    lq_buffer_t *buff;
    ranking_event_create(&buff, buffer_size);
    TestDataReader reader_data = {
        .dest = entries_dest,
        .destSize = source_size,
        .buff = buff,
    };
    TestDataWriter *writers_data = (TestDataWriter*)malloc(writers*sizeof(TestDataWriter));
    for (size_t i=0; i<writers; ++i) {
        writers_data[i].entries = &entries_source[i*params_per_thread];
        writers_data[i].entriesSize = params_per_thread;
        writers_data[i].buff = buff;
    }
    // perform the whole test here!
    // create reader
    for (size_t it=0; it<iterations; ++it) {
        int ret;
        pthread_t treader;
        pthread_t twriters[writers];
        ret = pthread_create(&treader, NULL, &read_batch, &reader_data);
        ASSERT_TRUE(ret == 0);
        // create writers
        for (size_t i=0; i<writers; ++i) {
            ret = pthread_create(&twriters[i], NULL, &write_batch, &writers_data[i]);
            ASSERT_TRUE(ret == 0);
        }
        // join writers
        for (size_t i=0; i<writers; ++i) {
            ret = pthread_join(twriters[i], NULL);
            ASSERT_TRUE(ret == 0);
        }
        // join reader
        ret = pthread_join(treader, NULL);
        ASSERT_TRUE(ret == 0);

        // check dest
        for (size_t i=0; i<source_size; ++i) {
            // check ith dest
            ASSERT_TRUE(entries_dest[i].type == EVENT_TOUCH);
            ASSERT_TRUE(entries_dest[i].data.touchData.address == (void*) i);
        }
        // clear dest
        memset(entries_dest, source_size, sizeof(EventEntry_t));
    }
    free(writers_data);
    ranking_event_destroy(buff);
    free(entries_dest);
    free(entries_source);
}

static void stress_tests_simple(void) {
    stress_test_simple(1, 10000000, 10000000, 20);
//     stress_test_simple(10, 1000000, 100000);
    stress_test_simple(10, 1000000, 1000000, 1);
//     stress_test_simple(3, 1000000, 100000);
//     stress_test_simple(2, 5000000, 10000000, 1);
}

TEST(LocklessRanking, LocklessStress){
    test_simple();
    test_simple_refill();
    stress_tests_simple();
}



#define struct_bar(size) typedef struct bar##size { char boo[(size)]; } bar##size

struct_bar(7);

#define test_slab_alloc(size, nof_elements) \
    do { \
        struct_bar(size); \
        slab_alloc_t temp; \
        int ret = slab_alloc_init(&temp, size, nof_elements); \
        ASSERT_TRUE(ret == 0 && "mutex creation failed!"); \
        slab_alloc_destroy(&temp); \
        ret = slab_alloc_init(&temp, size, nof_elements); \
        ASSERT_TRUE(ret == 0 && "mutex creation failed!"); \
        bar##size *elements[nof_elements]; \
        for (size_t i=0; i<nof_elements; ++i) { \
            elements[i] = (bar##size*)slab_alloc_malloc(&temp); \
            ASSERT_TRUE(elements[i] && "slab returned NULL!"); \
            memset(elements[i], i, size); \
        } \
        for (int i=0; i<nof_elements; ++i) { \
            for (size_t j=0; j<size; j++) \
                ASSERT_TRUE(elements[i]->boo[j] == (char)((unsigned)(i))%255); \
        } \
        ASSERT_TRUE(temp.used == nof_elements); \
        for (int i=0; i<nof_elements; ++i) { \
            slab_alloc_free(elements[i]); \
        } \
        ASSERT_TRUE(temp.used == nof_elements); \
        for (size_t i=0; i<nof_elements; ++i) { \
            elements[i] = (bar##size*)slab_alloc_malloc(&temp); \
            ASSERT_TRUE(elements[i] && "slab returned NULL!"); \
            memset(elements[i], i+15, size); \
        } \
        for (int i=0; i<nof_elements; ++i) { \
            for (size_t j=0; j<size; j++) \
                ASSERT_TRUE(elements[i]->boo[j] == (char)((unsigned)(i+15))%255); \
        } \
        ASSERT_TRUE(temp.used == nof_elements); \
        for (int i=0; i<nof_elements; ++i) { \
            slab_alloc_free(elements[i]); \
        } \
        ASSERT_TRUE(temp.used == nof_elements); \
        slab_alloc_destroy(&temp); \
    } while (0)

#define struct_bar_align(size) typedef struct bar_align##size { uint64_t boo[(size)]; } bar_align##size

struct_bar_align(7);

#define test_slab_alloc_alignment(size, nof_elements) \
    do { \
        struct_bar_align(size); \
        size_t bar_align_size=sizeof(bar_align##size); \
        slab_alloc_t temp; \
        int ret = slab_alloc_init(&temp, bar_align_size, nof_elements); \
        ASSERT_TRUE(ret == 0 && "mutex creation failed!"); \
        bar_align##size *elements[nof_elements]; \
        for (size_t i=0; i<nof_elements; ++i) { \
            elements[i] = (bar_align##size*)slab_alloc_malloc(&temp); \
            ASSERT_TRUE(elements[i] && "slab returned NULL!"); \
            for (size_t j=0; j<size; ++j) \
                elements[i]->boo[j] = i*nof_elements+j; \
        } \
        for (size_t i=0; i<nof_elements; ++i) { \
            for (size_t j=0; j<size; j++) \
                ASSERT_TRUE(elements[i]->boo[j] == i*nof_elements+j); \
        } \
        ASSERT_TRUE(temp.used == nof_elements); \
        for (size_t i=0; i<nof_elements; ++i) { \
            slab_alloc_free(elements[i]); \
        } \
        ASSERT_TRUE(temp.used == nof_elements); \
        for (size_t i=0; i<nof_elements; ++i) { \
            elements[i] = (bar_align##size*)slab_alloc_malloc(&temp); \
            ASSERT_TRUE(elements[i] && "slab returned NULL!"); \
            for (size_t j=0; j<size; ++j) \
                elements[i]->boo[j] = 7*i*nof_elements+j+5; \
        } \
        for (size_t i=0; i<nof_elements; ++i) { \
            for (size_t j=0; j<size; j++) \
                ASSERT_TRUE(elements[i]->boo[j] == 7*i*nof_elements+j+5); \
        } \
        ASSERT_TRUE(temp.used == nof_elements); \
        for (size_t i=0; i<nof_elements; ++i) { \
            slab_alloc_free(elements[i]); \
        } \
        ASSERT_TRUE(temp.used == nof_elements); \
        slab_alloc_destroy(&temp); \
    } while (0)

static void test_slab_alloc_static3(void) {
    struct_bar(3);
    size_t NOF_ELEMENTS=1024;
    size_t SIZE=3;
    slab_alloc_t temp;
    int ret = slab_alloc_init(&temp, SIZE, NOF_ELEMENTS);
    ASSERT_TRUE(ret == 0 && "slab alloc init failed!");
    slab_alloc_destroy(&temp);
    ret = slab_alloc_init(&temp, SIZE, NOF_ELEMENTS);
    ASSERT_TRUE(ret == 0 && "slab alloc init failed!");
    bar3 *elements[NOF_ELEMENTS];
    for (size_t i=0; i<NOF_ELEMENTS; ++i) {
        elements[i] = (bar3*)slab_alloc_malloc(&temp);
        ASSERT_TRUE(elements[i] && "slab returned NULL!");
        memset(elements[i], i, SIZE);
    }
    for (size_t i=0; i<NOF_ELEMENTS; ++i) {
        for (size_t j=0; j<SIZE; j++)
            ASSERT_TRUE(elements[i]->boo[j] == (char)((unsigned)(i))%255);
    }
    ASSERT_TRUE(temp.used == NOF_ELEMENTS);
    for (size_t i=0; i<NOF_ELEMENTS; ++i) {
        slab_alloc_free(elements[i]);
    }
    ASSERT_TRUE(temp.used == NOF_ELEMENTS);
    for (size_t i=0; i<NOF_ELEMENTS; ++i) {
        elements[i] = (bar3*)slab_alloc_malloc(&temp);
        ASSERT_TRUE(elements[i] && "slab returned NULL!");
        memset(elements[i], i+15, SIZE);
    }
    for (size_t i=0; i<NOF_ELEMENTS; ++i) {
        for (size_t j=0; j<SIZE; j++)
            ASSERT_TRUE(elements[i]->boo[j] == (char)((unsigned)(i+15))%255);
    }
    ASSERT_TRUE(temp.used == NOF_ELEMENTS);
    for (size_t i=0; i<NOF_ELEMENTS; ++i) {
        slab_alloc_free(elements[i]);
    }
    ASSERT_TRUE(temp.used == NOF_ELEMENTS);
    slab_alloc_destroy(&temp);
}

TEST(SlabAlloc, Basic) {
    test_slab_alloc_static3();
    test_slab_alloc(1, 1000000);
    test_slab_alloc(2, 1002300);
    test_slab_alloc(4, 798341);
    test_slab_alloc(8, 714962);
    test_slab_alloc(8, 1000000);
    test_slab_alloc(7, 942883);
    test_slab_alloc(17, 71962);
    test_slab_alloc(58, 214662);
}

TEST(SlabAlloc, Alignment) {
    test_slab_alloc_alignment(1, 100000);
    test_slab_alloc_alignment(1, 213299);
    test_slab_alloc_alignment(2, 912348);
    test_slab_alloc_alignment(4, 821429);
    test_slab_alloc_alignment(8, 814322);
    test_slab_alloc_alignment(7, 291146);
    test_slab_alloc_alignment(7, 291);
}

#define assert_close(a, b) do { \
const double ACCURACY=1e-9; /* arbitrary value */ \
    double diff = a-b; \
    double abs_diff = diff >= 0 ? diff : -diff; \
    ASSERT_LE(abs_diff, ACCURACY); \
} while(0)

TEST(RankingFixer, Basic) {
    ranking_info info;
    ranking_fixer_init_ranking_info(&info, 0.7, 1);
    double fixed_thresh;
    // corner cases:
    // ALL PMEM
    fixed_thresh = ranking_fixer_calculate_fixed_thresh(&info, 1);
    assert_close(fixed_thresh, 0);
    // ALL DRAM
    fixed_thresh = ranking_fixer_calculate_fixed_thresh(&info, 0);
    assert_close(fixed_thresh, 1);
    // already correct
    fixed_thresh = ranking_fixer_calculate_fixed_thresh(&info, 0.7);
    assert_close(fixed_thresh, 0.7);
    // 50%
    fixed_thresh = ranking_fixer_calculate_fixed_thresh(&info, 0.85);
    assert_close(fixed_thresh, 0.35);
    fixed_thresh = ranking_fixer_calculate_fixed_thresh(&info, 0.35);
    assert_close(fixed_thresh, 0.85);
    // 30%
    fixed_thresh = ranking_fixer_calculate_fixed_thresh(&info, 0.8);
    assert_close(fixed_thresh, 2./3.*0.7);
    fixed_thresh = ranking_fixer_calculate_fixed_thresh(&info, 2./3.*0.7);
    assert_close(fixed_thresh, 0.8);
}

TEST(RankingFixer, Gain) {
    ranking_info info;
    ranking_fixer_init_ranking_info(&info, 0.7, 2);
    double fixed_thresh;
    // corner cases:
    // ALL PMEM
    fixed_thresh = ranking_fixer_calculate_fixed_thresh(&info, 1);
    assert_close(fixed_thresh, -0.7);
    // ALL DRAM
    fixed_thresh = ranking_fixer_calculate_fixed_thresh(&info, 0);
    assert_close(fixed_thresh, 1.3);
    // already correct
    fixed_thresh = ranking_fixer_calculate_fixed_thresh(&info, 0.7);
    assert_close(fixed_thresh, 0.7);
    // 50%
    fixed_thresh = ranking_fixer_calculate_fixed_thresh(&info, 0.85);
    assert_close(fixed_thresh, 0);
    fixed_thresh = ranking_fixer_calculate_fixed_thresh(&info, 0.35);
    assert_close(fixed_thresh, 1);
    // 1/3
    fixed_thresh = ranking_fixer_calculate_fixed_thresh(&info, 0.8);
    assert_close(fixed_thresh, 1./3.*0.7);
    fixed_thresh = ranking_fixer_calculate_fixed_thresh(&info, 2./3.*0.7);
    assert_close(fixed_thresh, 0.9);
}
