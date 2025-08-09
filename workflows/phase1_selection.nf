#!/usr/bin/env nextflow

nextflow.enable.dsl=2

workflow PHASE1_SELECTION {
    take:
        frameworks_config
    
    main:
        // Validate framework configurations
        VALIDATE_FRAMEWORK_CONFIGS(frameworks_config)
        
        // Check framework availability
        CHECK_FRAMEWORK_AVAILABILITY(VALIDATE_FRAMEWORK_CONFIGS.out.valid_configs)
        
        // Select frameworks based on criteria
        SELECT_FRAMEWORKS(CHECK_FRAMEWORK_AVAILABILITY.out.available_frameworks)
        
        // Generate framework selection report
        GENERATE_SELECTION_REPORT(SELECT_FRAMEWORKS.out.selected_frameworks)
    
    emit:
        selected_frameworks = SELECT_FRAMEWORKS.out.selected_frameworks
        selection_report = GENERATE_SELECTION_REPORT.out.report
}

process VALIDATE_FRAMEWORK_CONFIGS {
    tag "Framework Configuration Validation"
    label 'cpu_intensive'
    
    input:
    path frameworks_config
    
    output:
    path "valid_configs.json"
    
    script:
    """
    python /app/scripts/validate_frameworks.py \\
        --config ${frameworks_config} \\
        --output valid_configs.json
    """
}

process CHECK_FRAMEWORK_AVAILABILITY {
    tag "Framework Availability Check"
    label 'cpu_intensive'
    
    input:
    path valid_configs
    
    output:
    path "available_frameworks.json"
    
    script:
    """
    python /app/scripts/check_availability.py \\
        --configs ${valid_configs} \\
        --output available_frameworks.json
    """
}

process SELECT_FRAMEWORKS {
    tag "Framework Selection"
    label 'cpu_intensive'
    
    input:
    path available_frameworks
    
    output:
    path "selected_frameworks.json"
    
    script:
    """
    python /app/scripts/select_frameworks.py \\
        --available ${available_frameworks} \\
        --output selected_frameworks.json \\
        --selection-criteria "maturity,performance,ecosystem"
    """
}

process GENERATE_SELECTION_REPORT {
    tag "Selection Report Generation"
    label 'cpu_intensive'
    
    input:
    path selected_frameworks
    
    output:
    path "framework_selection_report.md"
    
    script:
    """
    python /app/scripts/generate_selection_report.py \\
        --selected ${selected_frameworks} \\
        --output framework_selection_report.md
    """
} 