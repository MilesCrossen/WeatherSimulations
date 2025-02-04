import numpy as np
import matplotlib
import csv
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#capacity for 50 periodic functions
def y1(days):
    return (610.100261 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 123.046433 * np.cos(2 * np.pi * -0.002732 * days + 0.526557)
            + 123.046433 * np.cos(2 * np.pi * 0.002732 * days + -0.526557)
            + 35.649618 * np.cos(2 * np.pi * -0.051913 * days + -3.089786)
            + 35.649618 * np.cos(2 * np.pi * 0.051913 * days + 3.089786))

def y2(days):
    return (944.673951 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 409.421805 * np.cos(2 * np.pi * -0.002732 * days + 2.884963)
            + 409.421805 * np.cos(2 * np.pi * 0.002732 * days + -2.884963)
            + 37.916466 * np.cos(2 * np.pi * -0.005464 * days + -1.721603)
            + 37.916466 * np.cos(2 * np.pi * 0.005464 * days + 1.721603))

def y3(days):
    return (440.594246 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 116.128232 * np.cos(2 * np.pi * -0.002732 * days + 0.358378)
            + 116.128232 * np.cos(2 * np.pi * 0.002732 * days + -0.358378)
            + 19.283119 * np.cos(2 * np.pi * 0.117486 * days + -0.447141)
            + 19.283119 * np.cos(2 * np.pi * -0.117486 * days + 0.447141))


def y4(days):
    return (895.348824 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 388.655745 * np.cos(2 * np.pi * -0.002732 * days + 2.907815)
            + 388.655745 * np.cos(2 * np.pi * 0.002732 * days + -2.907815)
            + 29.011559 * np.cos(2 * np.pi * -0.005464 * days + -1.706297)
            + 29.011559 * np.cos(2 * np.pi * 0.005464 * days + 1.706297))
#ballyhaise solar

def y5(days):
    return (715.150580 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 169.444563 * np.cos(2 * np.pi * -0.002732 * days + 0.374241)
            + 169.444563 * np.cos(2 * np.pi * 0.002732 * days + -0.374241)
            + 37.523878 * np.cos(2 * np.pi * 0.142077 * days + 3.111961)
            + 37.523878 * np.cos(2 * np.pi * -0.142077 * days + -3.111961))
#carlow oakpark wind^3

def y6(days):
    return (947.698731 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 401.986193 * np.cos(2 * np.pi * -0.002732 * days + 2.905853)
            + 401.986193 * np.cos(2 * np.pi * 0.002732 * days + -2.905853)
            + 25.446340 * np.cos(2 * np.pi * -0.005464 * days + -1.528353)
            + 25.446340 * np.cos(2 * np.pi * 0.005464 * days + 1.528353))
#carlow oakpark solar

def y7(days):
    return (1103.861832 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 237.256946 * np.cos(2 * np.pi * -0.002732 * days + 0.450909)
            + 237.256946 * np.cos(2 * np.pi * 0.002732 * days + -0.450909)
            + 27.712610 * np.cos(2 * np.pi * 0.005464 * days + -1.914404)
            + 27.712610 * np.cos(2 * np.pi * -0.005464 * days + 1.914404))
#claremorris wind^3

def y8(days):
    return (903.445333 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 394.039971 * np.cos(2 * np.pi * -0.002732 * days + 2.892462)
            + 394.039971 * np.cos(2 * np.pi * 0.002732 * days + -2.892462)
            + 38.015242 * np.cos(2 * np.pi * 0.005464 * days + 1.725151)
            + 38.015242 * np.cos(2 * np.pi * -0.005464 * days + -1.725151))
#claremorris solar

def y9(days):
    return (846.175383 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 213.856719 * np.cos(2 * np.pi * -0.002732 * days + 0.395644)
            + 213.856719 * np.cos(2 * np.pi * 0.002732 * days + -0.395644)
            + 44.396759 * np.cos(2 * np.pi * -0.005464 * days + 0.902254)
            + 44.396759 * np.cos(2 * np.pi * 0.005464 * days + -0.902254))
#dunsany wind^3

def y10(days):
    return (343.309241 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 72.460247 * np.cos(2 * np.pi * -0.002732 * days + 0.670297)
            + 72.460247 * np.cos(2 * np.pi * 0.002732 * days + -0.670297)
            + 15.390016 * np.cos(2 * np.pi * 0.076503 * days + -0.760572)
            + 15.390016 * np.cos(2 * np.pi * -0.076503 * days + 0.760572))
#fermoy moorepark wind^3
def y11(days):
    return (935.938589 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 393.507642 * np.cos(2 * np.pi * -0.002732 * days + 2.918190)
            + 393.507642 * np.cos(2 * np.pi * 0.002732 * days + -2.918190)
            + 21.585158 * np.cos(2 * np.pi * 0.005464 * days + 1.392904) +
            21.585158 * np.cos(2 * np.pi * -0.005464 * days + -1.392904))
#fermoy moorepark solar
def y12(days):
    return (1939.404228 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 529.598164 * np.cos(2 * np.pi * -0.002732 * days + 0.219172)
            + 529.598164 * np.cos(2 * np.pi * 0.002732 * days + -0.219172)
            + 181.382104 * np.cos(2 * np.pi * -0.005464 * days + 0.534682)
            + 181.382104 * np.cos(2 * np.pi * 0.005464 * days + -0.534682))
#finner wind^3
def y13(days):
    return (951.031830 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
            409.147433 * np.cos(2 * np.pi * -0.002732 * days + 2.905366) +
            409.147433 * np.cos(2 * np.pi * 0.002732 * days + -2.905366) +
            31.293165 * np.cos(2 * np.pi * -0.005464 * days + -1.649791) +
            31.293165 * np.cos(2 * np.pi * 0.005464 * days + 1.649791))
#finner solar
def y14(days):
    return (915.591132 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
            209.872763 * np.cos(2 * np.pi * -0.002732 * days + 0.412902) +
            209.872763 * np.cos(2 * np.pi * 0.002732 * days + -0.412902) +
            44.180870 * np.cos(2 * np.pi * -0.084699 * days + -0.631869) +
            44.180870 * np.cos(2 * np.pi * 0.084699 * days + 0.631869))
#gurteen wdsp^3
def y15(days):
    return (947.711405 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 400.650933 * np.cos(2 * np.pi * -0.002732 * days + 2.897611)
            + 400.650933 * np.cos(2 * np.pi * 0.002732 * days + -2.897611)
            + 29.628462 * np.cos(2 * np.pi * 0.005464 * days + 1.753629)
            + 29.628462 * np.cos(2 * np.pi * -0.005464 * days + -1.753629))
#gurteen solar
def y16(days):
    return (954.588478 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 194.073008 * np.cos(2 * np.pi * -0.002732 * days + 0.230487)
            + 194.073008 * np.cos(2 * np.pi * 0.002732 * days + -0.230487)
            + 49.324582 * np.cos(2 * np.pi * -0.021858 * days + -0.667861)
            + 49.324582 * np.cos(2 * np.pi * 0.021858 * days + 0.667861))
#johnstown castle wdsp^3

def y17(days):
    return (1023.393226 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 442.266362 * np.cos(2 * np.pi * -0.002732 * days + 2.937333)
            + 442.266362 * np.cos(2 * np.pi * 0.002732 * days + -2.937333)
            + 29.143576 * np.cos(2 * np.pi * -0.005464 * days + -0.952498)
            + 29.143576 * np.cos(2 * np.pi * 0.005464 * days + 0.952498))
#johnstown castle solar
def y18(days):
    return (5008.804050 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 1326.813695 * np.cos(2 * np.pi * -0.002732 * days + -0.080705)
            + 1326.813695 * np.cos(2 * np.pi * 0.002732 * days + 0.080705)
            + 246.864158 * np.cos(2 * np.pi * -0.005464 * days + 0.179295)
            + 246.864158 * np.cos(2 * np.pi * 0.005464 * days + -0.179295))
#mace head wdsp^3
def y19(days):
    return (1019.889468 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 453.321877 * np.cos(2 * np.pi * -0.002732 * days + 2.883051)
            + 453.321877 * np.cos(2 * np.pi * 0.002732 * days + -2.883051)
            + 38.470130 * np.cos(2 * np.pi * -0.005464 * days + -1.558188)
            + 38.470130 * np.cos(2 * np.pi * 0.005464 * days + 1.558188))
#mace head solar
def y20(days):
    return (698.743597 * np.cos(2 * np.pi * 0.000000 * days + -0.000000) +
            146.495750 * np.cos(2 * np.pi * 0.128415 * days + 2.699013) +
            146.495750 * np.cos(2 * np.pi * -0.128415 * days + -2.699013) +
            142.697694 * np.cos(2 * np.pi * -0.117486 * days + -0.136537) +
            142.697694 * np.cos(2 * np.pi * 0.117486 * days + 0.136537))
def y21(days):
    return (889.008179 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 383.827537 * np.cos(2 * np.pi * -0.002732 * days + 2.896132)
            + 383.827537 * np.cos(2 * np.pi * 0.002732 * days + -2.896132)
            + 31.802652 * np.cos(2 * np.pi * -0.005464 * days + -1.712465)
            + 31.802652 * np.cos(2 * np.pi * 0.005464 * days + 1.712465))
#mount dillon solar
def y22(days):
    return (705.804733 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 189.272275 * np.cos(2 * np.pi * -0.002732 * days + 0.389232)
            + 189.272275 * np.cos(2 * np.pi * 0.002732 * days + -0.389232)
            + 26.993244 * np.cos(2 * np.pi * -0.005464 * days + 1.389271)
            + 26.993244 * np.cos(2 * np.pi * 0.005464 * days + -1.389271))
#mullingar wdsp^3
def y23(days):
    return (916.467203 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 392.937949 * np.cos(2 * np.pi * -0.002732 * days + 2.905887)
            + 392.937949 * np.cos(2 * np.pi * 0.002732 * days + -2.905887)
            + 31.580532 * np.cos(2 * np.pi * -0.005464 * days + -1.650745)
            + 31.580532 * np.cos(2 * np.pi * 0.005464 * days + 1.650745))
#mullingar solar
def y24(days):
    return (3002.236497 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 720.144850 * np.cos(2 * np.pi * -0.002732 * days + 0.186077)
            + 720.144850 * np.cos(2 * np.pi * 0.002732 * days + -0.186077)
            + 84.452642 * np.cos(2 * np.pi * 0.068306 * days + 1.347515) +
            84.452642 * np.cos(2 * np.pi * -0.068306 * days + -1.347515))
#roches point wdsp^3
def y25(days):
    return (1048.926846 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 439.458455 * np.cos(2 * np.pi * -0.002732 * days + 2.924591)
            + 439.458455 * np.cos(2 * np.pi * 0.002732 * days + -2.924591)
            + 27.974512 * np.cos(2 * np.pi * -0.005464 * days + -1.192798)
            + 27.974512 * np.cos(2 * np.pi * 0.005464 * days + 1.192798))
#roches point solar
def y26(days):
    return (3210.332005 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 953.364610 * np.cos(2 * np.pi * -0.002732 * days + 0.076167)
            + 953.364610 * np.cos(2 * np.pi * 0.002732 * days + -0.076167)
            + 171.070788 * np.cos(2 * np.pi * -0.174863 * days + 2.982949)
            + 171.070788 * np.cos(2 * np.pi * 0.174863 * days + -2.982949))
#sherkin island wdsp^3
def y27(days):
    return (1083.530563 * np.cos(2 * np.pi * 0.000000 * days + -0.000000)
            + 467.442574 * np.cos(2 * np.pi * -0.002732 * days + 2.937822)
            + 467.442574 * np.cos(2 * np.pi * 0.002732 * days + -2.937822)
            + 24.709511 * np.cos(2 * np.pi * -0.005464 * days + -0.987798)
            + 24.709511 * np.cos(2 * np.pi * 0.005464 * days + 0.987798))

#y11-y50 are currently fillers

#list of all basic functions
basis_functions = [
    y1, y2, y3, y4, y5, y6, y7, y8, y9, y10,
    y11, y12, y13, y14, y15, y16, y17, y18, y19, y20,
    y21, y22, y23, y24, y25, y26, y27
] #instead of manually calling each function, we can loop
#over this

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
#for each basis function, computes func(days) where days contains every dya of the year
#50x365 matrix
D = d(days) #demand for each day

#gradient descent parameters
learning_rate = 0.01 #base learning rate (delta x)
iterations = 50000 #more iterations = more accurate but requires more computing
#coefficients = np.ones(50) #initialise coefficients (y1...y50) to 1
coefficients = np.ones(27)

#we use Adaptive Movement Estimation (ADAM) to improve gradient descent
#https://blog.marketmuse.com/glossary/adaptive-moment-estimation-adam-definition/
m = np.zeros_like(coefficients) #Momentum term
v = np.zeros_like(coefficients) #RMSprop term
beta1 = 0.9 #momentum decay
#high momentum means the gradient is very smooth and depends quite heavily on past values
#we won't get sudden changes
beta2 = 0.999 #RMSprop decay
#RMSprop lowers learning rates if previous gradients are high, inreases them if they're low
#helps prevent over/undershoot
epsilon = 1e-8 #prevent division by 0 in RMSprop calculation

average_values = np.array([np.mean(func(days)) for func in basis_functions])
#averages values for each function are used later on

#gradient descent loop w/adam
for i in range(iterations):
    #predicted demand using current coefficients
    predicted = np.dot(coefficients, Y)
    #dot = dot product. Coefficients is 50x1 and Y  is 365x50 basis funcction outputs
    #so this creates a 365x1 output of predicted demand values

    #calculate error between predicted and real demand
    error = predicted - D

    #compute gradient
    gradient = np.dot(np.sign(error), Y.T) / len(days)
    #np.sign(error) yields 1 if predicted>D, -1 if predicted<D, and 0 if predicted=D
    #Y.T is transpose of Y just to make it work for multiplication.
    #this measures how much each basis veector contributes to overall prediction error

    lambda_penalty = 2 #if this parameter is higher, it penalises areas w/lower average
    #production more, causing the programme to recommend installations in areas with more
    #power production capacity
    gradient += lambda_penalty * coefficients / (average_values + 1e-6) #penalisation term
    #adds a penalty if average_values is low. Reduces coefficinets for low-energy areas


    #ADAM updating.
    m = beta1 * m + (1 - beta1) * gradient #momentum update using general formula
    v = beta2 * v + (1 - beta2) * (gradient ** 2) #RMSprop update using general formula
    m_hat = m / (1 - beta1 ** (i + 1)) #bias correction for momentum, prevents m from being
    #too small for early iterations
    v_hat = v / (1 - beta2 ** (i + 1)) #bias correction for RMSprop, prevents v from being
    #too small for early iterations

    #updating coefficients (final step of ADAM)
    coefficients -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    coefficients = np.maximum(coefficients, 0)

    #every 1k iterations, log and print progress
    if i % 1000 == 0:
        #coefficients = np.maximum(coefficients, 0)
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

#csv coefficient output file name
csv_filename = "coefficients.csv"

#find avg values before multiplication
#average_values = [np.mean(func(days)) for func in basis_functions]

#add to csv
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Basis Function", "Coefficient", "Average Value of Wind/Solar"]) #header contents
    for i, (coef, avg) in enumerate(zip(coefficients, average_values), start=1):
        writer.writerow([f"y{i}", coef, avg]) #write function name, coefficient, and average

print(f"Coefficients and average values saved to {csv_filename}")