"""
LineOptim High-Level API Example
===============================

This example demonstrates the high-level API of LineOptim with convenience functions.
"""

import lineoptim as lo

def main():
    """Demonstrate high-level LineOptim API."""
    
    print("LineOptim High-Level API Example")
    print("=" * 32)
    
    # Create a three-phase line with convenience function
    print("\n1. Creating three-phase distribution line...")
    main_line = lo.create_three_phase_line(
        name="Distribution Line A",
        voltage=400.0,
        resistivity=0.12  # ohm/km
    )
    
    # Add different types of loads using convenience functions
    print("2. Adding loads to the line...")
    lo.add_residential_load(main_line, "House 1", 100, 5.0)  # 5 kW at 100m
    lo.add_residential_load(main_line, "House 2", 200, 7.5)  # 7.5 kW at 200m
    lo.add_industrial_load(main_line, "Small Factory", 350, 25.0)  # 25 kW at 350m
    lo.add_residential_load(main_line, "House 3", 500, 4.8)  # 4.8 kW at 500m
    lo.add_industrial_load(main_line, "Workshop", 700, 12.0)  # 12 kW at 700m
    
    # Compute electrical parameters
    print("3. Computing electrical parameters...")
    main_line.recompute()
    
    # Get power summary
    print("4. Power summary:")
    power_summary = lo.get_power_summary(main_line)
    for key, value in power_summary.items():
        print(f"   {key}: {value:.2f}")
    
    # Calculate voltage drop
    print("\n5. Voltage analysis:")
    voltage_drop = lo.calculate_voltage_drop_percentage(main_line)
    max_voltage_drop = voltage_drop.max().item()
    print(f"   Maximum voltage drop: {max_voltage_drop:.2f}%")
    
    # Show voltage ranges
    print(f"   Voltage range: {main_line.voltage.min().min().item():.1f} - {main_line.voltage.max().max().item():.1f} V")
    
    # Optimize if voltage drop is too high
    if max_voltage_drop > 3.0:
        print("\n6. Optimizing line (voltage drop > 3%)...")
        
        # Use simple optimization interface
        results = lo.optimize_line_simple(
            main_line,
            max_voltage_drop=3.0,
            epochs=100,
            learning_rate=0.01
        )
        
        print("   Optimization results:")
        print(f"   Initial resistivity: {results['initial_resistivity'].mean().item():.4f} ohm/km")
        print(f"   Final resistivity: {results['final_resistivity'].mean().item():.4f} ohm/km")
        
        # Check new voltage drop
        new_voltage_drop = lo.calculate_voltage_drop_percentage(main_line)
        new_max_voltage_drop = new_voltage_drop.max().item()
        print(f"   New maximum voltage drop: {new_max_voltage_drop:.2f}%")
        print(f"   New voltage range: {main_line.voltage.min().min().item():.1f} - {main_line.voltage.max().max().item():.1f} V")
    
    # Create example network for comparison
    print("\n7. Creating example network for comparison...")
    example_line = lo.create_example_network()
    example_line.recompute()
    
    example_summary = lo.get_power_summary(example_line)
    print(f"   Example network total power: {example_summary['total_active_power_kw']:.1f} kW")
    
    # Visualization
    print("\n8. Plotting results...")
    try:
        # Plot voltage curve
        fig, ax = main_line.voltage.plot()
        fig.savefig('voltage_curve.png', dpi=150, bbox_inches='tight')
        print("   Voltage curve saved as 'voltage_curve.png'")
        
        # Plot network graph
        graph = lo.PlotGraph(main_line)
        graph.plot()
        graph.save('network_graph.pdf')
        print("   Network graph saved as 'network_graph.pdf'")
        
    except Exception as e:
        print(f"   Plotting skipped: {e}")
    
    # Save configuration
    print("\n9. Saving configuration...")
    main_line.save_to_json('high_level_example_config.json')
    print("   Configuration saved to 'high_level_example_config.json'")
    
    print("\n✓ High-level API example completed successfully!")


if __name__ == "__main__":
    main()