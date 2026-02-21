// =============================================================================
// paver_integration.cc - PAVER Integration Implementation
// =============================================================================
//
// Add this file to: src/gpgpu-sim/paver_integration.cc
// Add to Makefile: SRCS += paver_integration.cc
// =============================================================================

#include "paver_integration.h"
#include "paver_config.h"  // Auto-generated partition config
#include <fstream>
#include <sstream>
#include <iostream>

void paver_setup_kernel(kernel_info_t* kernel, unsigned num_sms) {
    if (!paver_should_enable(kernel)) {
        printf("PAVER: Disabled for kernel %u (too few TBs or no config)\n",
               kernel->get_uid());
        return;
    }
    
    // Convert static config arrays to runtime format
    std::vector<std::vector<unsigned>> partition_list;
    for (const auto& partition : PAVER_PARTITIONS) {
        partition_list.push_back(partition);
    }
    
    std::map<unsigned, std::vector<unsigned>> sm_assignments;
    for (const auto& kv : PAVER_SM_ASSIGNMENTS) {
        // Only include SMs that exist in the current config
        if (kv.first < num_sms) {
            sm_assignments[kv.first] = kv.second;
        }
    }
    
    // Handle case where config has fewer SMs than available
    // Distribute remaining partitions round-robin
    if (sm_assignments.size() < num_sms) {
        unsigned next_partition = partition_list.size();
        for (unsigned sm = sm_assignments.size(); sm < num_sms; sm++) {
            sm_assignments[sm] = std::vector<unsigned>();
        }
    }
    
    // Initialize PAVER runtime in kernel
    kernel->paver_init(partition_list, sm_assignments, PAVER_ENABLE_TASK_STEALING);
    
    printf("PAVER: Enabled for kernel %u (%s)\n",
           kernel->get_uid(), kernel->name().c_str());
    printf("PAVER:   Partitions: %zu\n", partition_list.size());
    printf("PAVER:   SMs: %zu\n", sm_assignments.size());
    printf("PAVER:   Task Stealing: %s\n", 
           PAVER_ENABLE_TASK_STEALING ? "enabled" : "disabled");
}

bool paver_load_config(
    const std::string& filename,
    std::vector<std::vector<unsigned>>& partitions,
    std::map<unsigned, std::vector<unsigned>>& sm_assign) {
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "PAVER: Could not open config file: " << filename << std::endl;
        return false;
    }
    
    partitions.clear();
    sm_assign.clear();
    
    std::string line;
    bool in_sm_section = false;
    
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            if (line.find("SM Assignments") != std::string::npos) {
                in_sm_section = true;
            }
            continue;
        }
        
        // Parse partition line: "PARTITION_ID: TB_ID, TB_ID, ..."
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) continue;
        
        unsigned id = std::stoul(line.substr(0, colon_pos));
        std::string tb_str = line.substr(colon_pos + 1);
        
        std::vector<unsigned> tbs;
        std::stringstream ss(tb_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            // Trim whitespace
            size_t start = token.find_first_not_of(" \t");
            size_t end = token.find_last_not_of(" \t");
            if (start != std::string::npos) {
                tbs.push_back(std::stoul(token.substr(start, end - start + 1)));
            }
        }
        
        if (!in_sm_section) {
            // This is a partition definition
            if (id >= partitions.size()) {
                partitions.resize(id + 1);
            }
            partitions[id] = tbs;
        }
    }
    
    // Generate default SM assignments (round-robin)
    // In production, this should be read from the config file
    unsigned num_partitions = partitions.size();
    unsigned num_sms = 15;  // Default, should be configurable
    
    for (unsigned sm = 0; sm < num_sms; sm++) {
        sm_assign[sm] = std::vector<unsigned>();
    }
    
    for (unsigned p = 0; p < num_partitions; p++) {
        unsigned sm = p % num_sms;
        sm_assign[sm].push_back(p);
    }
    
    return true;
}

bool paver_should_enable(kernel_info_t* kernel) {
    // PAVER is beneficial when there are enough TBs to partition
    size_t num_tbs = kernel->num_blocks();
    
    // Minimum TBs threshold (at least 2 TBs per SM)
    const size_t MIN_TBS_FOR_PAVER = 30;
    
    if (num_tbs < MIN_TBS_FOR_PAVER) {
        return false;
    }
    
    // Check if we have a valid PAVER config
    if (PAVER_PARTITIONS.empty()) {
        return false;
    }
    
    // Verify partition config matches kernel
    size_t config_tbs = 0;
    for (const auto& partition : PAVER_PARTITIONS) {
        config_tbs += partition.size();
    }
    
    if (config_tbs != num_tbs) {
        printf("PAVER: Warning - config has %zu TBs but kernel has %zu TBs\n",
               config_tbs, num_tbs);
        // Could still work if config has fewer TBs (remaining use LRR)
        // But if config has more TBs than kernel, that's a problem
        if (config_tbs > num_tbs) {
            return false;
        }
    }
    
    return true;
}

