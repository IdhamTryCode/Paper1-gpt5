#!/usr/bin/env nextflow

nextflow.enable.dsl=2

workflow PHASE4_BENCHMARK {
    take:
        validated_implementations
        environment_config
    
    main:
        // Execute classical ML benchmarks
        EXECUTE_CLASSICAL_ML_BENCHMARKS(validated_implementations, environment_config)
        
        // Execute deep learning benchmarks
        EXECUTE_DEEP_LEARNING_BENCHMARKS(validated_implementations, environment_config)
        
        // Execute reinforcement learning benchmarks
        EXECUTE_RL_BENCHMARKS(validated_implementations, environment_config)
        
        // Execute LLM benchmarks
        EXECUTE_LLM_BENCHMARKS(validated_implementations, environment_config)
        
        // Collect all benchmark results
        COLLECT_BENCHMARK_RESULTS(
            EXECUTE_CLASSICAL_ML_BENCHMARKS.out.results,
            EXECUTE_DEEP_LEARNING_BENCHMARKS.out.results,
            EXECUTE_RL_BENCHMARKS.out.results,
            EXECUTE_LLM_BENCHMARKS.out.results
        )
    
    emit:
        benchmark_results = COLLECT_BENCHMARK_RESULTS.out.all_results
}

process EXECUTE_CLASSICAL_ML_BENCHMARKS {
    tag "Classical ML Benchmarks"
    label 'cpu_intensive'
    
    input:
    path validated_implementations
    path environment_config
    
    output:
    path "classical_ml_results.json"
    
    script:
    """
    python /app/scripts/execute_classical_ml_benchmarks.py \\
        --implementations ${validated_implementations} \\
        --config ${environment_config} \\
        --output classical_ml_results.json
    """
}

process EXECUTE_DEEP_LEARNING_BENCHMARKS {
    tag "Deep Learning Benchmarks"
    label 'gpu_training'
    
    input:
    path validated_implementations
    path environment_config
    
    output:
    path "deep_learning_results.json"
    
    script:
    """
    python /app/scripts/execute_deep_learning_benchmarks.py \\
        --implementations ${validated_implementations} \\
        --config ${environment_config} \\
        --output deep_learning_results.json
    """
}

process EXECUTE_RL_BENCHMARKS {
    tag "Reinforcement Learning Benchmarks"
    label 'cpu_intensive'
    
    input:
    path validated_implementations
    path environment_config
    
    output:
    path "rl_results.json"
    
    script:
    """
    python /app/scripts/execute_rl_benchmarks.py \\
        --implementations ${validated_implementations} \\
        --config ${environment_config} \\
        --output rl_results.json
    """
}

process EXECUTE_LLM_BENCHMARKS {
    tag "LLM Benchmarks"
    label 'memory_intensive'
    
    input:
    path validated_implementations
    path environment_config
    
    output:
    path "llm_results.json"
    
    script:
    """
    python /app/scripts/execute_llm_benchmarks.py \\
        --implementations ${validated_implementations} \\
        --config ${environment_config} \\
        --output llm_results.json
    """
}

process COLLECT_BENCHMARK_RESULTS {
    tag "Benchmark Results Collection"
    label 'cpu_intensive'
    
    input:
    path classical_ml_results
    path deep_learning_results
    path rl_results
    path llm_results
    
    output:
    path "all_benchmark_results.json"
    
    script:
    """
    python /app/scripts/collect_benchmark_results.py \\
        --classical-ml ${classical_ml_results} \\
        --deep-learning ${deep_learning_results} \\
        --rl ${rl_results} \\
        --llm ${llm_results} \\
        --output all_benchmark_results.json
    """
} 