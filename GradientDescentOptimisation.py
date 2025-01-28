import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#capacity for 50 periodic functions
def y1(days):
    return (610.100261 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
            123.046433 * np.cos(2 * np.pi * -0.002732 * days + 0.526557) +
            123.046433 * np.cos(2 * np.pi * 0.002732 * days + -0.526557) +
            35.649618 * np.cos(2 * np.pi * -0.051913 * days + -3.089786) +
            35.649618 * np.cos(2 * np.pi * 0.051913 * days + 3.089786))
#athenry wind^3

def y2(days):
    return (944.467968 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
            409.621691 * np.cos(2 * np.pi * -0.002732 * days + 2.884843) +
            409.621691 * np.cos(2 * np.pi * 0.002732 * days + -2.884843) +
            37.762408 * np.cos(2 * np.pi * -0.005464 * days + -1.725207) +
            37.762408 * np.cos(2 * np.pi * 0.005464 * days + 1.725207))
#athenry solar

def y3(days):
    return (49.356367 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
            8.446029 * np.cos(2 * np.pi * -0.002732 * days + 0.368008) +
            8.446029 * np.cos(2 * np.pi * 0.002732 * days + -0.368008) +
            1.625258 * np.cos(2 * np.pi * -0.021858 * days + -0.770087) +
            1.625258 * np.cos(2 * np.pi * 0.021858 * days + 0.770087))
#ballyhaise wind^3

def y4(days):
    return (895.348824 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
            388.655745 * np.cos(2 * np.pi * -0.002732 * days + 2.907815) +
            388.655745 * np.cos(2 * np.pi * 0.002732 * days + -2.907815) +
            29.011559 * np.cos(2 * np.pi * -0.005464 * days + -1.706297) +
            29.011559 * np.cos(2 * np.pi * 0.005464 * days + 1.706297))
#ballyhaise solar

def y5(days):
    return (715.150580 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
            169.444563 * np.cos(2 * np.pi * -0.002732 * days + 0.374241) +
            169.444563 * np.cos(2 * np.pi * 0.002732 * days + -0.374241) +
            37.523878 * np.cos(2 * np.pi * 0.142077 * days + 3.111961) +
            37.523878 * np.cos(2 * np.pi * -0.142077 * days + -3.111961))
#carlow oakpark wind^3

def y6(days):
    return (947.698731 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
            401.986193 * np.cos(2 * np.pi * -0.002732 * days + 2.905853) +
            401.986193 * np.cos(2 * np.pi * 0.002732 * days + -2.905853) +
            25.446340 * np.cos(2 * np.pi * -0.005464 * days + -1.528353) +
            25.446340 * np.cos(2 * np.pi * 0.005464 * days + 1.528353))
#carlow oakpark solar

def y7(days):
    return (1103.444447 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
            237.021285 * np.cos(2 * np.pi * -0.002732 * days + 0.451714) +
            237.021285 * np.cos(2 * np.pi * 0.002732 * days + -0.451714) +
            27.924462 * np.cos(2 * np.pi * -0.005464 * days + 1.920502) +
            27.924462 * np.cos(2 * np.pi * 0.005464 * days + -1.920502))
#claremorris wind^3

def y8(days):
    return (505.948212 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
            217.824068 * np.cos(2 * np.pi * -0.002732 * days + 2.908963) +
            217.824068 * np.cos(2 * np.pi * 0.002732 * days + -2.908963) +
            24.502829 * np.cos(2 * np.pi * 0.005464 * days + 2.035442) +
            24.502829 * np.cos(2 * np.pi * -0.005464 * days + -2.035442))
#claremorris solar

def y9(days):
    return (76.143712 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
            12.194546 * np.cos(2 * np.pi * -0.002732 * days + 0.396359) +
            12.194546 * np.cos(2 * np.pi * 0.002732 * days + -0.396359) +
            2.272962 * np.cos(2 * np.pi * -0.005464 * days + 1.012486) +
            2.272962 * np.cos(2 * np.pi * 0.005464 * days + -1.012486))
#dunsany wind^3

def y10(days):
    return (951.031830 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
            409.147433 * np.cos(2 * np.pi * -0.002732 * days + 2.905366) +
            409.147433 * np.cos(2 * np.pi * 0.002732 * days + -2.905366) +
            31.293165 * np.cos(2 * np.pi * -0.005464 * days + -1.649791) +
            31.293165 * np.cos(2 * np.pi * 0.005464 * days + 1.649791))
#dunsany solar

def y11(days): return np.zeros_like(days)
def y12(days): return np.zeros_like(days)
def y13(days): return np.zeros_like(days)
def y14(days): return np.zeros_like(days)
def y15(days): return np.zeros_like(days)
def y16(days): return np.zeros_like(days)
def y17(days): return np.zeros_like(days)
def y18(days): return np.zeros_like(days)
def y19(days): return np.zeros_like(days)
def y20(days): return np.zeros_like(days)
def y21(days): return np.zeros_like(days)
def y22(days): return np.zeros_like(days)
def y23(days): return np.zeros_like(days)
def y24(days): return np.zeros_like(days)
def y25(days): return np.zeros_like(days)
def y26(days): return np.zeros_like(days)
def y27(days): return np.zeros_like(days)
def y28(days): return np.zeros_like(days)
def y29(days): return np.zeros_like(days)
def y30(days): return np.zeros_like(days)
def y31(days): return np.zeros_like(days)
def y32(days): return np.zeros_like(days)
def y33(days): return np.zeros_like(days)
def y34(days): return np.zeros_like(days)
def y35(days): return np.zeros_like(days)
def y36(days): return np.zeros_like(days)
def y37(days): return np.zeros_like(days)
def y38(days): return np.zeros_like(days)
def y39(days): return np.zeros_like(days)
def y40(days): return np.zeros_like(days)
def y41(days): return np.zeros_like(days)
def y42(days): return np.zeros_like(days)
def y43(days): return np.zeros_like(days)
def y44(days): return np.zeros_like(days)
def y45(days): return np.zeros_like(days)
def y46(days): return np.zeros_like(days)
def y47(days): return np.zeros_like(days)
def y48(days): return np.zeros_like(days)
def y49(days): return np.zeros_like(days)
def y50(days): return np.zeros_like(days)

#y11-y50 are currently fillers

#list of all basic functions
basis_functions = [
    y1, y2, y3, y4, y5, y6, y7, y8, y9, y10,
    y11, y12, y13, y14, y15, y16, y17, y18, y19, y20,
    y21, y22, y23, y24, y25, y26, y27, y28, y29, y30,
    y31, y32, y33, y34, y35, y36, y37, y38, y39, y40,
    y41, y42, y43, y44, y45, y46, y47, y48, y49, y50
]

#define demand function
def d(days):
    return (4454.708901 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
            202.979616 * np.cos(2 * np.pi * -0.002740 * days + 0.261259) +
            202.979616 * np.cos(2 * np.pi * 0.002740 * days + -0.261259) +
            133.571826 * np.cos(2 * np.pi * -0.142466 * days + -2.989836) +
            133.571826 * np.cos(2 * np.pi * 0.142466 * days + 2.989836))


#linear spacing of days in year
days = np.linspace(0, 365, 365)

#find calues of basis functions + demand function
Y = np.array([func(days) for func in basis_functions])
#for each basis function, computes fund(days) where days contains every dya of the year
#50x365 matrix
D = d(days) #demand for each day

#gradient descent parameters
learning_rate = 0.01 #base learning rate (delta x)
iterations = 50000 #more iterations = more accurate but requires more computing
coefficients = np.ones(50) #initialise coefficients (y1...y50) to 1

#we use Adaptive Movement Estimation (ADAM) to improve gradient descent
#https://blog.marketmuse.com/glossary/adaptive-moment-estimation-adam-definition/
# Adam optimizer parameters
m = np.zeros_like(coefficients) #Momentum term
v = np.zeros_like(coefficients) #RMSprop term
beta1 = 0.9 #omentum decay
beta2 = 0.999 #RMSprop decay
epsilon = 1e-8 #prevent division by 0

#gradient descent loop w/adam
for i in range(iterations):
    #predicted demand using current coefficients
    predicted = np.dot(coefficients, Y)
    #dot = dot product. Coefficients is 50x1 and Y  is 365x50
    #so this creates a 365x1 output of predicted demand values

    #calculate error
    error = predicted - D

    #compute gradient
    gradient = np.dot(np.sign(error), Y.T) / len(days)

    #ADAM updating.
    m = beta1 * m + (1 - beta1) * gradient #momentum update using general formula
    v = beta2 * v + (1 - beta2) * (gradient ** 2) #RMSprop update using general formula
    m_hat = m / (1 - beta1 ** (i + 1)) #bias correction for momentum
    v_hat = v / (1 - beta2 ** (i + 1)) #bias correction for RMSprop

    #updating coefficients
    coefficients -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    #every 1k iterations, log and print progress
    if i % 1000 == 0:
        absolute_error = np.sum(np.abs(error))
        print(f"Iteration {i}, Absolute Error: {absolute_error:.5f}")

#final coefficients
print("Final coefficients:", coefficients)

#plot only final production vs demand
plt.figure(figsize=(12, 6))
plt.plot(days, D, label="Demand (d(days))", linewidth=2, color="blue")
plt.plot(days, np.dot(coefficients, Y), label="Final Production", linestyle="--", linewidth=2, color="orange")

plt.legend()
plt.xlabel("Day of the Year")
plt.ylabel("Value")
plt.title("Final Production vs Demand")
plt.grid()
plt.show()