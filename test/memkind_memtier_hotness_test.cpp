// SPDX-License-Identifier: BSD-2-Clause
/* Copyright (C) 2021 Intel Corporation. */

#include <memkind/internal/memkind_memtier.h>
#include <memkind/internal/tachanka.h>

#include <random>
#include <thread>
#include <vector>
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
        assert(data != NULL);
        std::uniform_int_distribution<uint64_t> distr(0, size-1);
        size_t index = distr(generator);
        assert(index < size);
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

    // fill frequency array
    // objects with lower ID are more frequent
    // 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4 ....
    /*
    const int FREQ_ARRAY_LEN = (OBJS_NUM * OBJS_NUM + OBJS_NUM) / 2;
    int freq[FREQ_ARRAY_LEN];
    int aa = 0, bb = 0;
    for (it = 0; it < FREQ_ARRAY_LEN; it++) {
        freq[it] = aa;
        aa++;
        if (aa > bb) {
            bb++;
            aa = 0;
        }
    }
    */

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
            blocks[i].size=BLOCKS_SIZE-i;
            blocks[i].f=i;
            ranking_add(ranking, &blocks[i]);
        }

    }
    void TearDown()
    {
        ranking_destroy(ranking);
    }
};
constexpr size_t RankingTest::BLOCKS_SIZE;

TEST_F(RankingTest, check_hotness_highest) {
    double RATIO_PMEM_ONLY=0;
    double thresh_highest =
        ranking_calculate_hot_threshold_dram_total(
            ranking, RATIO_PMEM_ONLY);
    double thresh_highest_pmem =
        ranking_calculate_hot_threshold_dram_pmem(ranking, 0);
    ASSERT_EQ(thresh_highest, thresh_highest_pmem); // double for equality
    ASSERT_EQ(thresh_highest, BLOCKS_SIZE-1);
    ASSERT_EQ(thresh_highest, 99);
    for (size_t i=0; i<BLOCKS_SIZE-1; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
    ASSERT_EQ(ranking_is_hot(ranking, &blocks[BLOCKS_SIZE-1]), true);
}

TEST_F(RankingTest, check_hotness_lowest) {
    double RATIO_DRAM_ONLY=1;
    double thresh_lowest =
        ranking_calculate_hot_threshold_dram_total(
            ranking, RATIO_DRAM_ONLY);
    double thresh_lowest_pmem =
        ranking_calculate_hot_threshold_dram_pmem(
            ranking, std::numeric_limits<double>::max());
    ASSERT_EQ(thresh_lowest, thresh_lowest_pmem); // double for equality
    ASSERT_EQ(thresh_lowest, 0);
    for (size_t i=0; i<BLOCKS_SIZE; ++i) {
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
        ranking_calculate_hot_threshold_dram_total(ranking, RATIO_EQUAL);
    double thresh_equal_pmem =
        ranking_calculate_hot_threshold_dram_pmem(ranking, 1);
    ASSERT_EQ(thresh_equal, thresh_equal_pmem);
    ASSERT_EQ(thresh_equal, 29);
    for (size_t i=0; i<29; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
    for (size_t i=29; i<100; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), true);
    }
    ASSERT_EQ(BLOCKS_SIZE, 100u);
}

TEST_F(RankingTest, check_hotness_50_50_removed) {
    const size_t SUBSIZE=10u;
    for (size_t i=SUBSIZE; i<BLOCKS_SIZE; ++i) {
        ranking_remove(ranking, &blocks[i]);
    }
    double RATIO_EQUAL_TOTAL=0.5;
    double RATIO_EQUAL_PMEM=1;
    double thresh_equal = ranking_calculate_hot_threshold_dram_total(
        ranking, RATIO_EQUAL_TOTAL);
    double thresh_equal_pmem = ranking_calculate_hot_threshold_dram_pmem(
        ranking, RATIO_EQUAL_PMEM);
    // hand calculations:
    // 100, 99, 98, 97, 96, 95, 94, 93, 92, 91
    // sum:
    // 100, 199, 297, 394, 490 <- this is the one we are looking for
    ASSERT_EQ(thresh_equal, 4);
    ASSERT_EQ(thresh_equal, thresh_equal_pmem);
    for (size_t i=0; i<4; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
    for (size_t i=4; i<10; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), true);
    }
    ASSERT_EQ(BLOCKS_SIZE, 100u);
    ASSERT_EQ(SUBSIZE, 10u);
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
            blocks[i].size=BLOCKS_SIZE-i;
            blocks[i].f=i%50;
            ranking_add(ranking, &blocks[i]);
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
        ranking, RATIO_PMEM_ONLY_TOTAL);
    double thresh_highest_pmem = ranking_calculate_hot_threshold_dram_pmem(
        ranking, RATIO_PMEM_ONLY_PMEM);
    ASSERT_EQ(thresh_highest, (BLOCKS_SIZE-1)%50);
    ASSERT_EQ(thresh_highest, 49);
    ASSERT_EQ(thresh_highest, thresh_highest_pmem);
    for (size_t i=0; i<BLOCKS_SIZE-1; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), i==49);
    }
    ASSERT_EQ(ranking_is_hot(ranking, &blocks[BLOCKS_SIZE-1]), true);
}

TEST_F(RankingTestSameHotness, check_hotness_lowest) {
    double RATIO_DRAM_ONLY=1;
    double thresh_lowest = ranking_calculate_hot_threshold_dram_total(
        ranking, RATIO_DRAM_ONLY);
    double thresh_lowest_pmem = ranking_calculate_hot_threshold_dram_pmem(
        ranking, std::numeric_limits<double>::max());
    ASSERT_EQ(thresh_lowest, 0);
    ASSERT_EQ(thresh_lowest, thresh_lowest_pmem);
    for (size_t i=0; i<BLOCKS_SIZE; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), true);
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
        ranking, RATIO_EQUAL_TOTAL);
    double thresh_equal_pmem = ranking_calculate_hot_threshold_dram_pmem(
        ranking, RATIO_EQUAL_PMEM);
    ASSERT_EQ(thresh_equal, 19);
    ASSERT_EQ(thresh_equal, thresh_equal_pmem);
    for (size_t i=0; i<19; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
    for (size_t i=19; i<50; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), true);
    }
    for (size_t i=50; i<69; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
    for (size_t i=69; i<100; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), true);
    }
    ASSERT_EQ(BLOCKS_SIZE, 100u);
}

TEST_F(RankingTestSameHotness, check_hotness_50_50_removed) {
    const size_t SUBSIZE=10u;
    for (size_t i=SUBSIZE; i<BLOCKS_SIZE; ++i) {
        ranking_remove(ranking, &blocks[i]);
    }
    double RATIO_EQUAL_TOTAL=0.5;
    double RATIO_EQUAL_PMEM=1;
    double thresh_equal = ranking_calculate_hot_threshold_dram_total(
        ranking, RATIO_EQUAL_TOTAL);
    double thresh_equal_pmem = ranking_calculate_hot_threshold_dram_pmem(
        ranking, RATIO_EQUAL_PMEM);
    // hand calculations:
    // 100, 99, 98, 97, 96, 95, 94, 93, 92, 91
    // sum:
    // 100, 199, 297, 394, 490 <- this is the one we are looking for
    ASSERT_EQ(thresh_equal, 4);
    ASSERT_EQ(thresh_equal, thresh_equal_pmem);
    for (size_t i=0; i<4; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), false);
    }
    for (size_t i=4; i<10; ++i) {
        ASSERT_EQ(ranking_is_hot(ranking, &blocks[i]), true);
    }
    ASSERT_EQ(BLOCKS_SIZE, 100u);
    ASSERT_EQ(SUBSIZE, 10u);
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
    const size_t BUFF_SIZE = 2e8; // 200 MB
    const double frequency; /// @pre  0 < frequency <= 1
    double accumulated_freq=0;
    uint8_t *data;
    RandomIncremeneter incrementer;

public:
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
            for (volatile size_t i=0u; i<BUFF_SIZE; ++i) {
                sum += data[i];
            }
        }
        return sum;
    }

    memkind_t DetectKind() {
        return memkind_detect_kind(data);
    }

    double GetHotness() {
        return tachanka_get_addr_hotness(data);
    }

    Hotness_e GetHotnessType() {
        return tachanka_get_hotness_type(data);
    }

protected:
    virtual void FreeData() {
        memtier_free(data);
        data=nullptr;
    }

    virtual void AllocData(struct memtier_memory *m) {
        memtier_free(data);
        data = (uint8_t*)memtier_malloc(m, BUFF_SIZE);
        static size_t counter=0;
        std::string name = std::string("buff_")+std::to_string(counter)+"]";
        printf("allocated data [%s] at %p, size: [%lu]", name.c_str(), data, BUFF_SIZE);
        size_t str_size = strlen(name.c_str()) +1u;
        // use regular mallocs here - should not be an issue
        TouchCbArg_t *cb_arg = (TouchCbArg_t*)malloc(sizeof(TouchCbArg_t));
        cb_arg->name = (char*)malloc(sizeof(char)*str_size);
        cb_arg->counter = 0u;
        snprintf(cb_arg->name, str_size, "%s", name.c_str());
        int ret = tachanka_set_touch_callback(data, touch_cb, cb_arg);
        assert(ret==0);
        g_cbArgs.push_back(cb_arg);
        ++counter;
    }

    virtual void ReallocData(struct memtier_memory *m) {
        uint8_t *data_temp = (uint8_t*)memtier_malloc(m, BUFF_SIZE);
        memtier_free(data);
        data = data_temp;
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
        return TestBuffer::AllocData(m);
    }
    MEMKIND_NO_INLINE void ReallocData(struct memtier_memory *m) {
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
        return TestBuffer::AllocData(m);
    }
    MEMKIND_NO_INLINE void ReallocData(struct memtier_memory *m) {
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
        return TestBuffer::AllocData(m);
    }
    MEMKIND_NO_INLINE void ReallocData(struct memtier_memory *m) {
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

TEST_F(IntegrationHotnessSingleTest, test_random_hotness)
{
    // SIMPLE TEST - use only one Matrix per type
    TestBufferA &ma = bufferA[0];
    TestBufferB &mb = bufferB[0]; // should do half the work of ma
    auto start_point = std::chrono::steady_clock::now();
    auto end_point = start_point;
    double millis_elapsed=0;
    const double LIMIT_MILLIS=15000;
    size_t iterations=0u;
    for (iterations=0; millis_elapsed < LIMIT_MILLIS; ++iterations) {
        ma.DoSomeWork();
        mb.DoSomeWork();
        end_point = std::chrono::steady_clock::now();
        std::chrono::duration<double> duration = end_point-start_point;
        millis_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                .count();
    }
    double hotness_a=ma.GetHotness();
    double hotness_b=mb.GetHotness();
    size_t touches_a = g_cbArgs[0]->counter;
    size_t touches_b = g_cbArgs[1]->counter;

    double touch_ratio = ((double)touches_a)/touches_b;

    uint64_t asum = ma.CalculateSum();
    uint64_t bsum = mb.CalculateSum();
    // use calcualted data - prevent all loops from being optimized out
    printf("Total sums: A [%lu], B [%lu]\n", asum, bsum);
    printf("Total touches: A [%lu], B [%lu]\n", touches_a, touches_b);

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
    ASSERT_EQ(a_type, HOTNESS_HOT);
    ASSERT_EQ(b_type, HOTNESS_COLD);

    memkind_t a_kind = ma.DetectKind();
    memkind_t b_kind = mb.DetectKind();

    // check that both are on DRAM: initial allocation, hotness not known
    // at the beginning
    ASSERT_EQ(a_kind, MEMKIND_DEFAULT);
    ASSERT_EQ(b_kind, MEMKIND_DEFAULT);
}

// next test: run

TEST_F(IntegrationHotnessSingleTest, test_random_allocation_type)
{
    // SIMPLE TEST - use only one Matrix per type
    TestBufferB &mb = bufferB[0]; // should do half the work of ma
    TestBufferC &mc = bufferC[0]; // should do (ma_work+mb_work)/2
    TestBufferA &ma = bufferA[0];
    auto start_point = std::chrono::steady_clock::now();
    auto end_point = start_point;
    double millis_elapsed=0;
    const double LIMIT_MILLIS=15000;
    size_t iterations=0u;

    memkind_t a_kind = ma.DetectKind();
    memkind_t b_kind = mb.DetectKind();
    memkind_t c_kind = mc.DetectKind();

    // check that both are on DRAM: initial allocation, hotness not known
    // at the beginning
    ASSERT_EQ(a_kind, MEMKIND_DEFAULT);
    ASSERT_EQ(b_kind, MEMKIND_DEFAULT);
    ASSERT_EQ(c_kind, MEMKIND_DEFAULT);

    for (iterations=0; millis_elapsed < LIMIT_MILLIS; ++iterations) {
        ma.ReallocData(m_tier_memory);
        mb.ReallocData(m_tier_memory);
        mc.ReallocData(m_tier_memory);
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
    size_t touches_a = g_cbArgs[0]->counter;
    size_t touches_b = g_cbArgs[1]->counter;

    double touch_ratio = ((double)touches_a)/touches_b;

    uint64_t asum = ma.CalculateSum();
    uint64_t bsum = mb.CalculateSum();
    // use calcualted data - prevent all loops from being optimized out
    printf("Total sums: A [%lu], B [%lu]\n", asum, bsum);
    printf("Total touches: A [%lu], B [%lu]\n", touches_a, touches_b);

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

    double is_a_hot=ma.GetHotnessType();
    double is_b_hot=mb.GetHotnessType();
    ASSERT_EQ(is_a_hot, true);
    ASSERT_EQ(is_b_hot, false);

    ASSERT_EQ(a_kind, MEMKIND_DEFAULT); // more used kind was reallocated on DRAM
    ASSERT_EQ(b_kind, MEMKIND_REGULAR); // less used kind got reallocated on PMEM
    // TODO check where they are allocated - PMEM vs DRAM!
    // TODO make sure that elements persist - sum should be the same as that
    // of a reference, irrelevant object!!!
}
