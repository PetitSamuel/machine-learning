import math


def h(coef1, coef2, feature):
    # Returns the value for the model with provided coefs and feature
    return coef1 + coef2 * feature


def get_median(value):
    median = 0
    length = len(value)
    for i in range(length):
        median += value[i][0]
    return median / length


def cost_function(x, y, coef1, coef2):
    length = len(x)
    current_sum = 0
    for i in range(length):
        tmp = h(coef1, coef2, x[i][0]) - y[i][0]
        tmp = tmp * tmp
        current_sum += tmp
    return current_sum / length


def scaling_factor(value):
    # Compute the scaling factor for data standardisation
    # In this case the scaling factor is the std deviation
    length = len(value)
    valsSum = 0
    median = get_median(value)
    for i in range(length):
        valsSum += ((value[i][0] - median) ** 2) / length
    return math.sqrt(valsSum)


def gradient_descent(x, y, rate, iterations):
    # Find coefs that minimize our cost function using gradient descent
    coef1 = 0
    coef2 = 0
    length = len(x)
    deltas = []
    for _ in range(iterations):
        deltas.append(cost_function(x, y, coef1, coef2))
        sum1 = gradient_descent_sum(x, y, coef1, coef2, False)
        sum2 = gradient_descent_sum(x, y, coef1, coef2, True)
        gamma1 = ((-2 * rate) / length) * sum1
        gamma2 = ((-2 * rate) / length) * sum2
        coef1 += gamma1
        coef2 += gamma2
    return {
        "coef_1": coef1,
        "coef_2": coef2,
        "deltas": deltas,
    }


def gradient_descent_sum(x, y, coef1, coef2, multiply):
    # Computes the sum part for the gradient descent algorithm
    length = len(x)
    current_sum = 0
    for i in range(length):
        tmp = h(coef1, coef2, x[i][0]) - y[i][0]
        if multiply == True:
            tmp = tmp * x[i][0]
        current_sum += tmp
    return current_sum / length
