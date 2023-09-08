import pandas as pd
from sklearn.linear_model import LinearRegression
from fuzzywuzzy import fuzz
from datetime import datetime, timedelta


# Load the dataset
df = pd.read_excel('dataset.xlsx')

# Train your model here

# Example prediction function
def predict_expiry(category_name, purchase_date):
    # Find the closest matching category in the dataset
    df['Category_Score'] = df['Category'].apply(lambda x: fuzz.token_set_ratio(x, category_name))
    closest_match = df.loc[df['Category_Score'].idxmax()]

    # Extract the shelf life value from the closest match
    shelf_life = int(closest_match['Average Shelf Life(days)'])  # Convert to integer

    # Parse the purchase date input
    try:
        purchase_date = datetime.strptime(purchase_date, "%d %m %Y")
    except ValueError:
        print("Invalid date format. Please enter the date in the format: dd mm yyyy")
        return

    # Calculate expiry date
    expiry_date = purchase_date + timedelta(days=shelf_life)
    current_date = datetime.now()

    # Calculate remaining days until expiry
    remaining_days = (expiry_date - current_date).days

    if remaining_days < 0:
        return "Expired"
    else:
        return remaining_days

# Example usage
category_name = input("Enter the category name: ")
purchase_date = input("Enter the date of purchase (dd mm yyyy): ")

# Keep asking for a valid date until a valid one is provided
while True:
    result = predict_expiry(category_name, purchase_date)
    if result is not None:
        break
    purchase_date = input("Enter a valid date of purchase (dd mm yyyy): ")

if result == "Expired":
    print("Product has expired.")
else:
    print("Remaining days until expiry:", result)
