#ifndef PAVER_CONFIG_H
#define PAVER_CONFIG_H

#include <vector>
#include <map>

// Generated from paver_config.txt
// Method: kway_ts, 8 TBs, 1 partition

static const std::vector<std::vector<unsigned>> PAVER_PARTITIONS = {
    {6, 1, 0, 3, 2, 5, 4, 7}  // Partition 0
};

static const std::map<unsigned, std::vector<unsigned>> PAVER_SM_ASSIGNMENTS = {
    {0, {0}},   // SM 0 -> partition 0
    {1, {}},    // SM 1 -> no partitions
    {2, {}},
    {3, {}},
    {4, {}},
    {5, {}},
    {6, {}},
    {7, {}},
    {8, {}},
    {9, {}},
    {10, {}},
    {11, {}},
    {12, {}},
    {13, {}},
    {14, {}},
    {15, {}},
    {16, {}},
    {17, {}},
    {18, {}},
    {19, {}},
    {20, {}},
    {21, {}},
    {22, {}},
    {23, {}},
    {24, {}},
    {25, {}},
    {26, {}},
    {27, {}},
    {28, {}},
    {29, {}},
    {30, {}},
    {31, {}},
    {32, {}},
    {33, {}},
    {34, {}},
    {35, {}},
    {36, {}},
    {37, {}},
    {38, {}},
    {39, {}},
    {40, {}},
    {41, {}},
    {42, {}},
    {43, {}},
    {44, {}},
    {45, {}},
    {46, {}},
    {47, {}},
    {48, {}},
    {49, {}},
    {50, {}},
    {51, {}},
    {52, {}},
    {53, {}},
    {54, {}},
    {55, {}},
    {56, {}},
    {57, {}},
    {58, {}},
    {59, {}},
    {60, {}},
    {61, {}},
    {62, {}},
    {63, {}},
    {64, {}},
    {65, {}},
    {66, {}},
    {67, {}},
    {68, {}},
    {69, {}},
    {70, {}},
    {71, {}},
    {72, {}},
    {73, {}},
    {74, {}},
    {75, {}},
    {76, {}},
    {77, {}},
    {78, {}},
    {79, {}}
};

static const bool PAVER_ENABLE_TASK_STEALING = true;

#endif // PAVER_CONFIG_H