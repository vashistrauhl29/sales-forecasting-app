import pandas as pd

# Create a template DataFrame
df = pd.DataFrame({
    "Year": [2020, 2020, 2020, 2020],
    "Quarter": [1, 2, 3, 4],
    "Time Period": [1, 2, 3, 4],
    "Actual Sales": [200, 300, 500, 600]
})

# Save to Excel
df.to_excel("template_sales_data.xlsx", index=False)
print("Template saved as template_sales_data.xlsx")

