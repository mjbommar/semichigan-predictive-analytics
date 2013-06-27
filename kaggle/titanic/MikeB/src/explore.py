# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# Imports
import pandas
import pylab
import sklearn

# <codecell>

# Define some utilities
def column_to_indicator(data, column_name):
    '''
    Return a modified dataframe with a column converted to indicators.
    '''
    values = data[column_name].unique()
    for value in values:
        # Skip NaNs
        if value in ['nan', 'NaN', nan]:
            continue
            
        # Otherwise, build and put
        indicator_name = '{}_{}'.format(column_name, value)
        data[indicator_name] = (data[column_name] == value).apply(int)

    return data

def count_cabins(cabin_value):
    try:
        return len(cabin_value.split())
    except:
        return 0

def get_first_char(value):
    try:
        return value.strip()[0]
    except:
        return None

def get_cabins(cabin_value):
    '''
    Return a list of cabins from cabin string.
    '''
    if type(cabin_value) != str:
        return []
    else:
        return str(cabin_value).split()

def get_cabin_map(cabin_value):
    '''
    Return cabin letter and number midpoint for each cabin letter.
    '''
    # Skip empty cabin values
    if type(cabin_value) != str:
        return {}
    
    # Build a temporary map with all numbers
    cabin_map_temp = {}
    for value in get_cabins(cabin_value):
        # Get letter and number
        letter = value[0]
        if len(value) > 1:
            number = int(value[1:])
        else:
            number = None
        
        # Add to temp map
        if letter in cabin_map_temp:
            cabin_map_temp[letter].append(number)
        else:
            cabin_map_temp[letter] = [number]
            
    # Loop over found cabin letters to get average
    cabin_map = {}
    for letter in cabin_map_temp:
        try:
            number_sum = sum(cabin_map_temp[letter])
            cabin_map[letter] = int(float(number_sum) / len(cabin_map_temp[letter]))
        except Exception, E:
            cabin_map[letter] = None
    
    # Return average cabin map numbers
    return cabin_map

def get_all_cabin_letters(data):
    '''
    Get all cabin numbers.
    '''
    return sorted(list(set([x[0] for x in get_all_cabins(data)])))

def get_all_cabins(data):
    '''
    Get all cabins.
    '''
    cabin_set = set()
    for value in data['cabin']:
        cabin_set.update(get_cabins(value))
    
    return sorted(list(cabin_set))

# <codecell>

# Define data preprocessing method
def preprocess_data(data):
    '''
    Return a preprocessed data frame.
    '''
    # Add cabin letter/number features
    cabin_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
    cabin_indicator_columns = {}
    cabin_number_columns = {}
    for letter in cabin_letters:
        cabin_indicator_columns[letter] = []
        cabin_number_columns[letter] = []
        
    for cabin_map in data['cabin'].apply(get_cabin_map):
        # Map all letters
        for letter in cabin_letters:
            # Handle indicator and average
            if letter in cabin_map:
                cabin_indicator_columns[letter].append(1)
                cabin_number_columns[letter].append(cabin_map[letter])
            else:
                cabin_indicator_columns[letter].append(0)
                cabin_number_columns[letter].append(None)
    
    # Now put columns into dataframe
    for letter in cabin_letters:
        data['cabin_' + letter] = cabin_indicator_columns[letter]
        #data['cabin_number_' + letter] = cabin_number_columns[letter]
    
    # Map embarked to indicator
    data = column_to_indicator(data, 'embarked')
    data = column_to_indicator(data, 'sex')
    
    # Add some name indicator variables
    has_given_name = []
    has_miss = []
    has_mr = []
    has_mrs = []
    has_given = []
    has_rev = []
    for value in data['name']:
        # Check for some strings
        if 'Mrs.' in value:
            has_mrs.append(1)
        else:
            has_mrs.append(0)
            
        if 'Mr.' in value:
            has_mr.append(1)
        else:
            has_mr.append(0)
            
        if 'Miss.' in value:
            has_miss.append(1)
        else:
            has_miss.append(0)
        
        if 'Rev.' in value:
            has_rev.append(1)
        else:
            has_rev.append(0)
        
        if '(' in value:
            has_given.append(1)
        else:
            has_given.append(0)
            
    data['has_miss'] = has_miss
    data['has_mr'] = has_mr
    data['has_mrs'] = has_mrs
    data['has_given'] = has_given
    data['has_rev'] = has_rev
    
    # Get ticket start
    data['ticket_char'] = data['ticket'].apply(get_first_char)
    data = column_to_indicator(data, 'ticket_char')
        
    # Binarize family variables
    data['has_family_peer'] = (data['sibsp'] > 0).apply(int)
    data['has_family_nonpeer'] = (data['parch'] > 0).apply(int)
    data['has_all_family'] = data['has_family_peer'] * data['has_family_nonpeer']
    data['cabin_tokens'] = data['cabin'].apply(count_cabins)
    data['ticket_count'] = data['ticket'].apply(len)
    data['name_tokens'] = data['cabin'].apply(count_cabins)
    data['name_count'] = data['name'].apply(len)
    #print [x for x in data.columns]
    
    # Split into features and targets
    target_columns = ['survived']
    ignore_columns = ['name', 'ticket', 'embarked', 'cabin', 'sex']
    feature_columns = ['pclass', 'age', 'sibsp', 'parch', 'fare',
        'cabin_tokens', 'ticket_count', 'name_tokens', 'name_count',
        'cabin_A', 'cabin_B', 'cabin_C', 'cabin_D', 'cabin_E', 'cabin_F', 'cabin_G', 'cabin_T', 
        'embarked_C', 'embarked_S', 'embarked_Q', 
        'sex_female', 'sex_male', 
        'has_miss', 'has_mr', 'has_mrs', 'has_given', 'has_rev',
        'ticket_char_3', 'ticket_char_2', 'ticket_char_A', 'ticket_char_S', 'ticket_char_P', 'ticket_char_C', 'ticket_char_1',
        'has_family_peer', 'has_family_nonpeer', 'has_all_family']
    #feature_columns = 
    normalize_columns = feature_columns#['pclass', 'age', 'fare', 'sibsp', 'parch']
    
    data_features = data[feature_columns]
    
    try:
        data_target = data[target_columns]
    except:
        data_target = None            
    
    # Scale and handle NAs
    for column_name in data_features.columns:
        if column_name in normalize_columns:
            mean = data_features[column_name].mean()
            sd = data_features[column_name].std()
            data_features[column_name] = (data_features[column_name] - mean) / sd
            data_features[column_name].fillna(0, inplace=True)
        else:
            data_features[column_name].fillna(data_features[column_name].mean(), inplace=True)
    
    return (data_features, data_target)

# <codecell>

# Load data
data_train = pandas.read_csv('data/train.csv')
data_test_kaggle = pandas.read_csv('data/test.csv')

# Split into test and train
test_ratio = 0.3
test_rows = random.choice(data_train.index, int(test_ratio * len(data_train.index)))
data_test = data_train.ix[test_rows]
data_train = data_train.drop(test_rows)

# Preprocess
data_train_features, data_train_target = preprocess_data(data_train)
data_test_features, data_test_target = preprocess_data(data_test)
data_test_kaggle_features, data_test_kaggle_target = preprocess_data(data_test_kaggle)

# <codecell>

# Scratch
print(get_all_cabin_letters(data_train))

# <codecell>

# Visualize some basic information
exclude_columns = ['name', 'ticket', 'cabin']

for column_name in data_train.columns:
    # Skip excluded values
    if column_name in exclude_columns:
        continue
    
    # Determine column type
    column_type = data_train.dtypes[column_name]
    
    # Output some summary info
    print("{}: {}".format(column_name, column_type))
    
    # Depending on type, plot a histogram.
    
    if column_type in [int64]:
        f = pylab.figure()
        data_train[column_name].hist()
        pylab.title('{} histogram'.format(column_name))
    elif column_type in [object]:
        f = pylab.figure()
        data_train[column_name].value_counts().plot(kind='bar')
        pylab.title('{} histogram'.format(column_name))

# <codecell>

# Visualize some basic information
exclude_columns = ['name', 'ticket', 'cabin', 'survived']

for column_name in data_train.columns:
    # Skip excluded values
    if column_name in exclude_columns:
        continue
    
    # Determine column type
    column_type = data_train.dtypes[column_name]
    
    # Output some summary info
    print("{}: {}".format(column_name, column_type))
    
    # Depending on type, plot a histogram.
    
    if column_type in [int64]:
        f = plt.figure()
        plt.scatter(data_train[column_name], data_train['survived'], alpha=0.025)
        f.suptitle('{} histogram'.format(column_name))
    elif column_type in [object]:
        pass

# <codecell>


