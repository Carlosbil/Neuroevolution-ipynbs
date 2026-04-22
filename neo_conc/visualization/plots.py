"""
Visualization module for evolution progress and results.

Provides functions to visualize:
- Fitness evolution across generations
- Population diversity (standard deviation)
- Detailed statistics and convergence analysis
- Failure analysis for debugging
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_fitness_evolution(neuroevolution, config: dict = None):
    """
    Plots fitness evolution across generations.

    Args:
        neuroevolution: HybridNeuroevolution instance with generation_stats
        config: Optional config dict for fitness_threshold line
    """
    if not neuroevolution.generation_stats:
        print("WARNING: No statistics data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data and filter 0.00 fitness
    generations = []
    avg_fitness = []
    max_fitness = []
    min_fitness = []
    std_fitness = []
    
    for stat in neuroevolution.generation_stats:
        # Only include if valid fitness (> 0.00)
        if stat['max_fitness'] > 0.00:
            generations.append(stat['generation'])
            avg_fitness.append(stat['avg_fitness'])
            max_fitness.append(stat['max_fitness'])
            min_fitness.append(stat['min_fitness'])
            std_fitness.append(stat['std_fitness'])
    
    if not generations:
        print("WARNING: No valid fitness data to plot (all are 0.00)")
        return
    
    # Graph 1: Fitness evolution
    ax1.plot(generations, max_fitness, 'g-', linewidth=2, marker='o', label='Maximum Fitness')
    ax1.plot(generations, avg_fitness, 'b-', linewidth=2, marker='s', label='Average Fitness')
    ax1.plot(generations, min_fitness, 'r-', linewidth=2, marker='^', label='Minimum Fitness')
    ax1.fill_between(generations, 
                     [max(0, avg - std) for avg, std in zip(avg_fitness, std_fitness)],
                     [avg + std for avg, std in zip(avg_fitness, std_fitness)],
                     alpha=0.2, color='blue')
    
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness (%)')
    ax1.set_title('Fitness Evolution by Generation (Excluding 0.00%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add target fitness line if config provided
    if config and 'fitness_threshold' in config:
        ax1.axhline(y=config['fitness_threshold'], color='orange', linestyle='--', 
                    label=f"Target ({config['fitness_threshold']}%)")
        ax1.legend()
    
    # Set Y axis limits for better visualization
    y_min = max(0, min(min_fitness) - 5)
    y_max = min(100, max(max_fitness) + 5)
    ax1.set_ylim(y_min, y_max)
    
    # Graph 2: Diversity (standard deviation)
    ax2.plot(generations, std_fitness, 'purple', linewidth=2, marker='D')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness Standard Deviation')
    ax2.set_title('Population Diversity (Excluding 0.00%)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Show additional information
    print(f"Plotted data:")
    print(f"   Generations with valid fitness: {len(generations)}")
    print(f"   Best fitness achieved: {max(max_fitness):.2f}%")
    print(f"   Final average fitness: {avg_fitness[-1]:.2f}%")
    if len(generations) < len(neuroevolution.generation_stats):
        excluded = len(neuroevolution.generation_stats) - len(generations)
        print(f"   WARNING: Excluded generations (0.00 fitness): {excluded}")


def show_evolution_statistics(neuroevolution, config: dict = None):
    """
    Shows detailed evolution statistics.

    Args:
        neuroevolution: HybridNeuroevolution instance with generation_stats
        config: Optional config dict for target fitness info
    """
    print("DETAILED EVOLUTION STATISTICS")
    print("="*60)
    
    if not neuroevolution.generation_stats:
        print("WARNING: No statistics available")
        return
    
    # Filter statistics with valid fitness
    valid_stats = [stat for stat in neuroevolution.generation_stats if stat['max_fitness'] > 0.00]
    
    if not valid_stats:
        print("WARNING: No valid statistics (all fitness are 0.00)")
        return
    
    final_stats = valid_stats[-1]
    
    print(f"Completed generations: {neuroevolution.generation}")
    print(f"Generations with valid fitness: {len(valid_stats)}")
    if len(valid_stats) < len(neuroevolution.generation_stats):
        excluded = len(neuroevolution.generation_stats) - len(valid_stats)
        print(f"WARNING: Generations with 0.00 fitness (excluded): {excluded}")
    
    print(f"\nFINAL STATISTICS (excluding 0.00 fitness):")
    print(f"   Final best fitness: {final_stats['max_fitness']:.2f}%")
    print(f"   Final average fitness: {final_stats['avg_fitness']:.2f}%")
    print(f"   Final minimum fitness: {final_stats['min_fitness']:.2f}%")
    print(f"   Final standard deviation: {final_stats['std_fitness']:.2f}%")
    
    # Progress across generations
    if len(valid_stats) > 1:
        initial_max = valid_stats[0]['max_fitness']
        final_max = valid_stats[-1]['max_fitness']
        improvement = final_max - initial_max
        
        print(f"\nPROGRESS:")
        print(f"   Initial fitness: {initial_max:.2f}%")
        print(f"   Final fitness: {final_max:.2f}%")
        print(f"   Total improvement: {improvement:.2f}%")
        if initial_max > 0:
            print(f"   Relative improvement: {(improvement/initial_max)*100:.1f}%")
    
    # Convergence analysis
    print(f"\nCONVERGENCE CRITERIA:")
    if config and 'fitness_threshold' in config:
        if neuroevolution.best_individual and neuroevolution.best_individual['fitness'] >= config['fitness_threshold']:
            print(f"   ✅ Target fitness reached ({config['fitness_threshold']}%)")
        else:
            print(f"   ❌ Target fitness NOT reached ({config['fitness_threshold']}%)")
    
    if config and 'max_generations' in config:
        if neuroevolution.generation >= config['max_generations']:
            print(f"   ⏱️ Maximum generations reached ({config['max_generations']})")
    
    # Additional performance statistics
    all_max_fitness = [stat['max_fitness'] for stat in valid_stats]
    all_avg_fitness = [stat['avg_fitness'] for stat in valid_stats]
    
    print(f"\nGENERAL STATISTICS:")
    print(f"   Best fitness of entire evolution: {max(all_max_fitness):.2f}%")
    print(f"   Average fitness of entire evolution: {np.mean(all_avg_fitness):.2f}%")
    print(f"   Average improvement per generation: {(max(all_max_fitness) - min(all_max_fitness))/len(valid_stats):.2f}%")
    
    if neuroevolution.best_individual:
        print(f"\nBest individual ID: {neuroevolution.best_individual['id']}")
        print(f"Best individual fitness: {neuroevolution.best_individual['fitness']:.2f}%")


def analyze_failed_evaluations(neuroevolution):
    """
    Analyzes evaluations that resulted in 0.00 fitness.

    Args:
        neuroevolution: HybridNeuroevolution instance with generation_stats
    """
    print("\nFAILED EVALUATIONS ANALYSIS")
    print("="*50)
    
    total_generations = len(neuroevolution.generation_stats)
    failed_generations = len([stat for stat in neuroevolution.generation_stats if stat['max_fitness'] == 0.00])
    
    if failed_generations == 0:
        print("✅ No failed evaluations (0.00 fitness)")
        return
    
    success_rate = ((total_generations - failed_generations) / total_generations) * 100
    
    print(f"Failure summary:")
    print(f"   Total generations: {total_generations}")
    print(f"   Failed generations: {failed_generations}")
    print(f"   Success rate: {success_rate:.1f}%")
    
    if failed_generations > 0:
        failed_gens = [stat['generation'] for stat in neuroevolution.generation_stats if stat['max_fitness'] == 0.00]
        print(f"   Generations with failures: {failed_gens}")
        
        print(f"\nPossible causes of 0.00 fitness:")
        print(f"   • Errors in model architecture")
        print(f"   • Memory problems (GPU/RAM)")
        print(f"   • Invalid hyperparameter configurations")
        print(f"   • Errors during training")


def configure_plot_style():
    """Configure matplotlib and seaborn style for consistent plots."""
    plt.style.use('default')
    sns.set_palette("husl")
