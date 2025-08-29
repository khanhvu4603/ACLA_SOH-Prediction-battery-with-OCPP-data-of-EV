import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# Đọc kết quả inference
df = pd.read_csv("soh_predictions_mc.csv")

x = df["soh_true"].values
y = df["soh_pred_mean"].values

# Fit hồi quy tuyến tính y = a*x + b
coef = np.polyfit(x, y, 1)
a, b = coef
y_fit = a * x + b

# Tính R²
r2 = r2_score(x, y)   # R² giữa true và pred
r2_line = r2_score(y, y_fit)  # R² của đường fit (thực ra gần như giống)

plt.figure(figsize=(7,7))
plt.scatter(x, y, alpha=0.5, label="Predictions")
# Vẽ đường hồi quy
plt.plot(x, y_fit, 'r-', linewidth=2, label=f"Fit: y={a:.2f}x+{b:.2f}, R²={r2:.3f}")
# Vẽ đường y=x (lý tưởng)
plt.plot([x.min(), x.max()], [x.min(), x.max()], 'g--', label="Ideal y=x")

plt.xlabel("SOH True (proxy)")
plt.ylabel("SOH Predicted")
plt.title("Scatter SOH_pred vs SOH_true (with regression line)")
plt.legend()
plt.grid(True)
plt.show()
