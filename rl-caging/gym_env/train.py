from rl_caging_env import RLCagingEnv

# Initialize environment
env = RLCagingEnv()
obs, infos = env.reset()

# Action space bounds
linear_low, linear_high = 0.0, 0.5
angular_low, angular_high = -2.0, 2.0

# Run multiple simulation steps
for step in range(100):  # you can adjust total steps
    # Random actions for all agents
    actions = {
        agent: env.action_space(agent).sample()
        for agent in env.agents
    }

    # Step environment
    obs, rewards, terminations, truncations, infos = env.step(actions)

    print(f"\n=== Step {step + 1} ===")
    print("Actions:")
    for agent, action in actions.items():
        print(f"  {agent}: linear={action[0]:.3f} m/s, angular={action[1]:.3f} rad/s")

    print("\n=== Agent Observations ===")
    for agent_name, agent_data in infos.items():
        print(f"\n▶ {agent_name.upper()}")
        print(f"  Object:")
        print(f"    Distance: {agent_data['object']['distance']} m")
        print(f"    Angle:    {agent_data['object']['angle']}°")

        print(f"  Target:")
        print(f"    Distance: {agent_data['target']['distance']} m")
        print(f"    Angle:    {agent_data['target']['angle']}°")

        print("  Other Agents:")
        for key, val in agent_data.items():
            if key not in ["object", "target"]:
                print(f"    {key.capitalize()}:")
                print(f"      Distance: {val['distance']} m")
                print(f"      Angle:    {val['angle']}°")

    print("\n=== Rewards ===")
    for agent, reward in rewards.items():
        print(f"  {agent}: {reward:.3f}")

    # ✅ Check if any agent terminated (collision or condition)
    if any(terminations.values()):
        print("\n⚠️ Termination detected — resetting environment...\n")
        obs, infos = env.reset()

print("\nSimulation complete.")
