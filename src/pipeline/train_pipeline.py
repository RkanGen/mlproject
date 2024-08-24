import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression  # You can change this to your preferred model

from src.exception import CustomException
from src.utils import save_object

class TrainPipeline:
    def __init__(self):
        self.training_data_path = os.path.join('artifacts', 'train.csv')
        self.test_data_path = os.path.join('artifacts', 'test.csv')
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

    def initiate_data_ingestion(self):
        try:
            # Read the dataset
            df = pd.read_csv('notebook\data\stud.csv')
            
            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save train and test sets
            train_set.to_csv(self.training_data_path, index=False, header=True)
            test_set.to_csv(self.test_data_path, index=False, header=True)
            
            return (
                self.training_data_path,
                self.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        try:
            train_df = pd.read_csv(self.training_data_path)
            test_df = pd.read_csv(self.test_data_path)
            
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            numerical_columns = ['reading_score', 'writing_score']
            
            # Define the custom ranking for each ordinal variable
            gender_categories = ['male', 'female']
            race_categories = ['group A', 'group B', 'group C', 'group D', 'group E']
            parental_education_categories = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", "master's degree"]
            lunch_categories = ['standard', 'free/reduced']
            test_prep_categories = ['none', 'completed']
            
            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
                ]
            )
            
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array, preprocessor):
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            if test_model_score < 0.6:
                raise CustomException("Model is not good fit as test score is less than 0.6")
            
            save_object(
                file_path=self.model_path,
                obj=model
            )
            
            save_object(
                file_path=self.preprocessor_path,
                obj=preprocessor
            )
            
            return (train_model_score, test_model_score)
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_train_pipeline(self):
        try:
            train_path, test_path = self.initiate_data_ingestion()
            preprocessor = self.initiate_data_transformation()
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            target_column_name = 'math_score'  # Assuming 'math_score' is the target variable
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            train_arr = pd.concat([pd.DataFrame(input_feature_train_arr), target_feature_train_df], axis=1).values
            test_arr = pd.concat([pd.DataFrame(input_feature_test_arr), target_feature_test_df], axis=1).values
            
            train_model_score, test_model_score = self.initiate_model_trainer(train_arr, test_arr, preprocessor)
            
            print(f"Training Score: {train_model_score}")
            print(f"Testing Score: {test_model_score}")
            
            return (train_model_score, test_model_score)
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    train_pipeline = TrainPipeline()
    train_pipeline.initiate_train_pipeline()