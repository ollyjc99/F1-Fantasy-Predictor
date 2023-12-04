import pandas as pd
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler

class Utilities:

    def format_time(x):
        if not any(i in x for i in ['DNF', 'DNS']):
            if ':' in x:
                return round(float(str(x).split(':')[1]) + (60 * float(str(x).split(':')[0])), 3) if x != 0 else 0
            else:
                return(round(float(x), 3))
        else:
            return x
        
    def reverse_name(x):
        if str(x) == 'guanyu_zhou':
            name = x.split('_')
            return name[1] + '_' + name[0]
        else:
            return x
    
    def score_regression(N, df, model, params_to_drop):
        score = 0
        for circuit in df[df.season == N]['round'].unique():

            test = df[(df.season == N) & (df['round'] == circuit)]
            X_test = test.drop(params_to_drop, axis=1)
            y_test = test.driver_points_from

            #scaling
            X_test = pd.DataFrame(StandardScaler().transform(X_test), columns = X_test.columns)

            # make predictions
            prediction_df = pd.DataFrame(model.predict(X_test), columns=['predicted_points'])
            prediction_df['actual_points'] = y_test.reset_index(drop=True)

            prediction_df['predicted_winner'] = prediction_df.predicted_points.map(lambda x: 1 if x == prediction_df.predicted_points.max() else 0)
            prediction_df['actual_winner'] = prediction_df.actual_points.map(lambda x: 1 if x == prediction_df.actual_points.max() else 0)

            score += precision_score(prediction_df.actual_winner, prediction_df.predicted_winner)

        return score / df[df.season == N]['round'].nunique()