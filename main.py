import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# Memuat data dari Kaggle
data = pd.read_csv("Student_Performance.csv")


# Problem 1: Durasi waktu belajar (TB) terhadap nilai ujian (NT)
TB = data["Hours Studied"].values.reshape(-1, 1)
NT = data["Performance Index"].values

# Metode 1: Model Linear
linear_model = LinearRegression()
linear_model.fit(TB, NT)
NT_pred_linear = linear_model.predict(TB)

# Metode 3: Model Eksponensial
def exponential_model(x, a, b):
    return a * np.exp(b * x)

popt_exponential, _ = curve_fit(exponential_model, TB.flatten(), NT)
NT_pred_exponential = exponential_model(TB.flatten(), *popt_exponential)

# Metode Opsional: Model Laju Pertumbuhan Jenuh
def logistic_growth_model(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

popt_logistic, _ = curve_fit(logistic_growth_model, TB.flatten(), NT, p0=[max(NT), 1, np.median(TB)])
NT_pred_logistic = logistic_growth_model(TB.flatten(), *popt_logistic)

# Menghitung galat RMS
rms_linear = np.sqrt(mean_squared_error(NT, NT_pred_linear))
rms_exponential = np.sqrt(mean_squared_error(NT, NT_pred_exponential))
rms_logistic = np.sqrt(mean_squared_error(NT, NT_pred_logistic))

print(f"RMS Error - Model Linear: {rms_linear}")
print(f"RMS Error - Model Eksponensial: {rms_exponential}")
print(f"RMS Error - Model Laju Pertumbuhan Jenuh: {rms_logistic}")

# Plot hasil regresi
plt.figure(figsize=(14, 7))

# Plot model linear
plt.subplot(1, 3, 1)
plt.scatter(TB, NT, label='Data asli', color='blue')
plt.plot(TB, NT_pred_linear, label='Regresi Linear', color='red')
plt.xlabel('Durasi waktu belajar (TB)')
plt.ylabel('Nilai ujian (NT)')
plt.title('Model Linear')
plt.legend()

# Plot model eksponensial
plt.subplot(1, 3, 2)
plt.scatter(TB, NT, label='Data asli', color='blue')
plt.plot(TB, NT_pred_exponential, label='Regresi Eksponensial', color='red')
plt.xlabel('Durasi waktu belajar (TB)')
plt.ylabel('Nilai ujian (NT)')
plt.title('Model Eksponensial')
plt.legend()

# Plot model laju pertumbuhan jenuh
plt.subplot(1, 3, 3)
plt.scatter(TB, NT, label='Data asli', color='blue')
plt.plot(TB, NT_pred_logistic, label='Regresi Laju Pertumbuhan Jenuh', color='red')
plt.xlabel('Durasi waktu belajar (TB)')
plt.ylabel('Nilai ujian (NT)')
plt.title('Model Laju Pertumbuhan Jenuh')
plt.legend()

plt.tight_layout()
plt.show()
