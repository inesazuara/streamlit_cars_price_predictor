import pandas as pd

# Loading dataset for training
def load_data(file_path,sep):
       return pd.read_csv(file_path, sep=sep)

# print(df_cars.head())

def preprocess_data(df):
       # Selecting variables for the model and diveding by numeric and categorical
       cols = ['make', 'model', 'months_old', 'power', 'sale_type',
              'num_owners', 'gear_type', 'fuel_type', 'kms', 'price']

       cat_cols = ['make', 'model', 'sale_type', 'gear_type', 'fuel_type']

       # Creating dummies
       df = pd.get_dummies(df[cols],
                           prefix_sep='_',
                           drop_first=True,
                           columns=cat_cols)

       # Cleaning nulls
       # First cleaning num_owners nulls converting in categorical variable and creating Null category
       df['num_owners'] = df['num_owners'].astype('object')
       filter_num_owner = df['num_owners'] >= 3
       df.loc[filter_num_owner,'num_owners'] = '3+'

       df = pd.get_dummies(df,prefix_sep='_',
                           dummy_na=True,
                           drop_first=True,
                           columns=['num_owners'])


       return df

if __name__ == "__main__":
       df_cars = load_data("data/coches_usados_esp.csv",";")
       df_cars_clean = preprocess_data(df_cars)
       # print(df_cars_clean)
       print(df_cars_clean)
       print(df_cars_clean[['months_old', 'power', 'kms', 'price']].isna().sum())