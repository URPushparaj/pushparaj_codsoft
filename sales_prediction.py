# Sales Prediction With Conditional User Input & Comparison

# Predefined industry average sales data
industry_average = 4000
good_sales_threshold = industry_average * 1.2  # 20% above average
bad_sales_threshold = industry_average * 0.8   # 20% below average

# Function to calculate average sales
def calculate_average(sales_list):
    total = 0
    count = 0
    for sale in sales_list:
        total += sale
        count += 1
    return total / count if count > 0 else 0

# Get ad spend from user
while True:
    try:
        ad_spend = float(input("Enter advertising spend: "))
        if ad_spend < 0:
            print("❌ Advertising spend cannot be negative. Try again.")
        else:
            break
    except ValueError:
        print("❌ Invalid input! Please enter a valid number.")

# Get season factor with validation
while True:
    try:
        season = int(input("Enter season factor (1=Low, 2=Medium, 3=High): "))
        if season in [1, 2, 3]:
            break
        else:
            print("❌ Invalid choice! Please enter 1, 2, or 3.")
    except ValueError:
        print("❌ Invalid input! Please enter a number (1, 2, or 3).")

# Ask if the user wants to enter past sales manually or use predefined data
use_predefined = input("Do you want to use predefined past sales data? (yes/no): ").strip().lower()

if use_predefined == "yes":
    past_sales = [3200, 4000, 4500, 3800, 4200]
    print("✅ Using predefined past sales data:", past_sales)
else:
    past_sales = []
    print("Enter past sales data (Enter -1 to stop):")
    while True:
        try:
            sale = float(input("Sales: "))
            if sale == -1:
                break
            elif sale < 0:
                print("❌ Sales cannot be negative. Try again.")
            else:
                past_sales.append(sale)
        except ValueError:
            print("❌ Invalid input! Please enter a valid number.")

# Get market growth factor
while True:
    try:
        market_growth = float(input("Enter market growth factor (percentage increase, e.g., 5 for 5%): "))
        break
    except ValueError:
        print("❌ Invalid input! Please enter a valid number.")

# Calculate average past sales
past_sales_avg = calculate_average(past_sales)

# Sales prediction formula
predicted_sales = (ad_spend * 6) + (season * 150) + (past_sales_avg * 0.5) + (market_growth * 2)

# Sales classification
if predicted_sales >= good_sales_threshold:
    sales_status = "Good Sales 🚀"
elif predicted_sales <= bad_sales_threshold:
    sales_status = "Bad Sales ❌"
else:
    sales_status = "Average Sales 🙂"

# Display results
print("\n--- Sales Prediction Result ---")
print("Predicted Sales:", predicted_sales)
print("Industry Average Sales:", industry_average)
print("Sales Performance:", sales_status)
