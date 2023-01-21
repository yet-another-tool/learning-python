def dynamic_learning_rate(learning_rate, cost_change, treshold, effect=0.1):
    print(f"Learning Rate: {learning_rate}")
    if(cost_change >= treshold and cost_change > 0):
        print("Don't change anything")
    elif(cost_change < 0):
        print("It is getter worse, we are increasing, reduce learning rate value 🎓🎓🎓")
        learning_rate -= round(float(learning_rate*effect), 6)
    elif(cost_change > 0):
        print("It is getter better, we are decreasing, increase learning rate value 🔨🔨")
        learning_rate += round(float(learning_rate*(effect/4)), 6)

    print(f"Updated Learning Rate: {learning_rate}")
    return learning_rate
