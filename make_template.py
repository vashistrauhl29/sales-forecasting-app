import pandas as pd

# Create updated DataFrame based on the new structure
data = {
    "Year": [
        2017, 2017, 2017, 2017,
        2018, 2018, 2018, 2018,
        2019, 2019, 2019, 2019,
        2020, 2020, 2020, 2020,
        2021, 2021, 2021, 2021,
        2022, 2022, 2022, 2022,
        2023, 2023, 2023, 2023
    ],
    "Quarter": [
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4
    ],
    "Time Period": list(range(1, 29)),
    "Actual Sales": [
        684.20, 584.10, 765.38, 892.28,
        885.40, 759.25, 840.22, 910.46,
        900.22, 875.90, 920.84, 988.52,
        915.78, 880.66, 944.78, 1004.32,
        960.44, 934.78, 990.84, 1050.66,
        1010.24, 976.52, 1045.30, 1100.44,
        1065.22, 1030.34, 1104.66, 1172.10
    ]
}

df = pd.DataFrame(data)

# Save to Excel
df.to_excel("template_sales_data.xlsx", index=False)
print("✅ Updated template saved as template_sales_data.xlsx")

