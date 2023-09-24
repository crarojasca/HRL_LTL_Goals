def fourrooms_transfer_experiment(env, agent, logger, cfg, run_name):
    # Train
    print("First Stage")
    env.spec.end_state = 1
    agent.run(env, logger)
    agent.save(cfg, run_name+"_FirstStage")

    print("Second Stage")
    # Update Goal
    env.spec.end_state = 2
    # Clear Buffer
    agent.buffer.clear()
    # Set New Number of Max Episodes
    agent.max_episodes = cfg.agent.max_episodes*2
    # Run
    agent.run(env, logger)
    agent.save(cfg, run_name+"_SecondStage")

    print("Third Stage")
    env.spec.end_state = 3
    # Clear Buffer
    agent.buffer.clear()
    # Set New Number of Max Episodes
    agent.max_episodes = cfg.agent.max_episodes*3
    # Run
    agent.run(env, logger)
    agent.save(cfg, run_name+"_ThirdStage")