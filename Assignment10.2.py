'''In this assignment, students will be using the K-nearest neighbors algorithm to predict
how many points NBA players scored in the 2013-2014 season.
A look at the data
Before we dive into the algorithm, letâ€TMs take a look at our data. Each row in the data
contains information on how a player performed in the 2013-2014 NBA season.
Download 'nba_2013.csv' file from this link:
https://www.dropbox.com/s/b3nv38jjo5dxcl6/nba_2013.csv?dl=0
Here are some selected columns from the data:
player - name of the player
pos - the position of the player
g - number of games the player was in
gs - number of games the player started
pts - total points the player scored
There are many more columns in the data, mostly containing information about average
player game performance over the course of the season. See this site for an explanation
of the rest of them.
We can read our dataset in and figure out which columns are present:

import pandas
with open("nba_2013.csv", 'r') as csvfile:
nba = pandas.read_csv(csvfile)

NOTE:​ ​The​ ​solution​ ​shared​ ​through​ ​Github​ ​should​ ​contain​ ​the​ ​source​ ​code​ ​used​ ​and​'''


try:
    import pandas
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split
    
    with open("nba_2013.csv", 'r') as csvfile:
        nba = pandas.read_csv(csvfile,usecols=['player','pos','g','gs','pts'])
        
    X = nba.loc[:,['pos','g','gs']]
    y= nba.loc[:,'pts']
    
    #obj_col = list(nba.select_dtypes(include=['object']).columns)
    num_col = list(X.select_dtypes(include=['int64','float64']).columns)
    
    #One hot encoding
    X=pandas.get_dummies(X,dummy_na=False,drop_first=True,columns=['pos'])
    
    #Splitting dataset
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=0)
    
    #KNN algorithm    
    regressor= KNeighborsRegressor()
    regressor.fit(X_train,y_train)
    
    #Prediction
    y_pred= regressor.predict(X_test)
    
    #Check Accuracy Score
    regressor.score(X_train,y_train)
    regressor.score(X_test,y_test)

except Exception as e:
    print(e)
