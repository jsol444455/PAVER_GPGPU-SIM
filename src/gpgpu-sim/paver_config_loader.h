// =============================================================================
// paver_config_loader.h
// =============================================================================
// Add this file to: src/gpgpu-sim/paver_config_loader.h
//
// Reads paver_config.txt and calls kernel->paver_init()
// =============================================================================

#ifndef PAVER_CONFIG_LOADER_H
#define PAVER_CONFIG_LOADER_H

#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class paver_config_loader {
public:
    bool enabled;
    std::string method;
    bool task_stealing;
    std::vector<std::vector<unsigned>> partitions;
    std::map<unsigned, std::vector<unsigned>> sm_assignments;

    paver_config_loader() : enabled(false), task_stealing(true) {}

    bool load(const std::string& filepath) {
        enabled = false;
        partitions.clear();
        sm_assignments.clear();

        std::ifstream file(filepath);
        if (!file.is_open()) {
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            std::string keyword;
            iss >> keyword;

            if (keyword == "paver_enabled") {
                int val; iss >> val;
                enabled = (val == 1);
            }
            else if (keyword == "paver_method") {
                iss >> method;
            }
            else if (keyword == "paver_task_stealing") {
                int val; iss >> val;
                task_stealing = (val == 1);
            }
            else if (keyword == "paver_partition") {
                unsigned partition_id;
                std::string tb_str;
                iss >> partition_id >> tb_str;

                std::vector<unsigned> tb_list;
                if (!tb_str.empty()) {
                    std::stringstream ss(tb_str);
                    std::string token;
                    while (std::getline(ss, token, ',')) {
                        tb_list.push_back(std::stoul(token));
                    }
                }

                if (partition_id >= partitions.size()) {
                    partitions.resize(partition_id + 1);
                }
                partitions[partition_id] = tb_list;
            }
            else if (keyword == "paver_sm_assign") {
                unsigned sm_id;
                std::string partition_str;
                iss >> sm_id;

                std::vector<unsigned> partition_ids;
                if (iss >> partition_str && !partition_str.empty()) {
                    std::stringstream ss(partition_str);
                    std::string token;
                    while (std::getline(ss, token, ',')) {
                        partition_ids.push_back(std::stoul(token));
                    }
                }
                sm_assignments[sm_id] = partition_ids;
            }
        }
        file.close();

        if (enabled) {
            printf("PAVER: Loaded config - method=%s, partitions=%zu, task_stealing=%d\n",
                   method.c_str(), partitions.size(), task_stealing);
        }
        return enabled;
    }
};

#endif // PAVER_CONFIG_LOADER_H
