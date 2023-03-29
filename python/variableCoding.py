
try:
  from google.colab import drive
  drive.mount('/content/drive/')
  shared_drive_foldername = 'NUTRECON'
  root = '/content/drive/Shareddrives/{}/'.format(shared_drive_foldername)
  dataPath_ = root + '/2. Data/raw/nutrecon/'
  # dataPath_ = = root + '/2. Data/raw/nutrecon_psych/'
  print('Running Code in Colab')
except:
  root = 'D:/FV/Projects/NUTRECON/nutreconDrive/'
  dataPath_ = "D:/FV/Projects/NUTRECON/Data/nutrecon/"
  # dataPath = root + "2. demoData/raw/nutrecon/"
  print('Running Code locally')

# Taste Strips IDs
sour = [1,2,3,4]
salt = [6,7,8,9]
sweet = [11,12,13,14]
bitter = [15,16,17,18]
H2O = [5]


# create a class contaning the attributes necessary for single subject analysis
class Vars:
  
  dataPath = dataPath_
  sequences_dataPath = dataPath + 'sequences/'
  responses_dataPath = dataPath + 'responses/'
  yogurtPrep_path = root + '0. Yogurt Preparation/yogurt prep.xlsx'
  if 'demo' in dataPath:
    experiment_code = 'exampleSub'
  else:
    experiment_code = 'nutre'
  

  # dictionary containing a code for all the flavors tested
  flavorCodes = {
      'blueberry' : 'g',
      'cashew' : 'c', 
      'dragon fruit': 'h',
      'ginseng' : 'i',
      'grapefruit': 'j',
      'licorice': 'k',
      'lychee': 'd', 
      'pomegranate': 'e' 
    }

  # dictionary containing the correspondance between image number and shape
  imageCodes = {
      'Image I' : 'square',
      'Image II' : 'hexagon',
      'Image V' : 'circle',
      'Image VI' : 'cross'
      }

  # dictionary with index - image number correspondance
  imageDecoder = {
      0: 'Image I',
      1: 'Image II',
      #2: 'Image III',
      #3: 'Image IV',
      4: 'Image V',
      5: 'Image VI'
      }

  # file identifiers and column names for dataframes
  pres_order_colName = 'Presentation Order'
  flavorName_colName = 'Flavor'
  flavorID_colName = 'flavor_id'

  orders_id = '_presOrder_'
  pres_order_fileID = 'day1_presOrder'
  pres_order_d2_fileID = 'day2_presOrder'
  pres_order_d3_fileID = 'day3_presOrder'

  ratings_fileID = '_ratings_'
  flav_ratings_fileID = 'day1_rating'
  flav_ratings_d2_fileID = 'day2_rating'
  flav_ratings_d3_fileID = 'day3_rating'

  intensity_colName = 'intensity'
  novelty_colName = 'novelty'
  pleasanteness_colName = 'pleasantness'

  learn_order_fileID = 'day1_learnOrder'
  learningOrder_colName = 'Learning order'
  shapeName_colName = 'shape'
  shapeID_colName = 'shape_id'
  shapeRomanID_colName = 'shapeRoman_id'
  # imageName_colName = 'shape'
  # imageID_colName = 'shape_id'
  # imageRomanID_colName = 'shapeRoman_id'

  associaitonTest_fileID = '_atest_'
  associaitonOrder_fileID = '_atestOrder_'

  assocTestOrder1_colName = 'Day 1 - Association Test order'
  assoc1_fileID = 'day1_atest'
  assoc1_order_fileID = 'day1_atestOrder'

  assocTestOrder2_colName = 'Day 2 - Association Test order'
  assoc2_fileID = 'day2_atest'
  assoc2_order_fileID = 'day2_atestOrder'

  assocTestOrder3_colName = 'Day 3 - Association Test order'
  assoc3_fileID = 'day3_atest'
  assoc3_order_fileID = 'day3_atestOrder'

  conditioning_order_fileID = '_condOrder_'
  conditioning_order_colName = 'Conditioning Order'

  # files containing the order of the trials in the neuroeconomics task performed at day 2 and 3
  neuroEconOrder0_fileID = 'day1_neuroEconOrder'
  neuroEconOrder1_fileID = 'day2_neuroEconOrder'
  neuroEconOrder2_fileID = 'day3_neuroEconOrder'
  
  # files containing the users' responses to the neuroeconomics task
  neuroEcon_id = '_neuroEcon_'
  neuroEcon_d1_responses_fileID = 'day1_neuroEcon'
  neuroEcon_d2_responses_fileID = 'day2_neuroEcon'
  neuroEcon_d3_responses_fileID = 'day3_neuroEcon'

  # trials that were actually realized after the task was completed
  neurEconRealization_d1_fileID = 'day1_neurEconRealization'
  neurEconRealization_d2_fileID = 'day2_neurEconRealization'
  neurEconRealization_d3_fileID = 'day3_neurEconRealization'

  # Probability of choosing Lottery colName
  probLotteryChoice_colName = 'Prob. of choosing lottery'

  # files containing users' responses to Taste Strips
  tasteStrips_fileID = '_tasteStrips_'
  stripsOrder_colName = 'order_id'
  stripID_colName = 'strip_id'
  stripName_colName = 'strip'
  tastant_colName = 'tastant'

  # dictionary containing Taste Strips information
  taste_stripsID_orders = {
    1: sour + H2O + salt + H2O + sweet + bitter,
    2: salt + H2O + sour + H2O + sweet + bitter,
    3: sweet + H2O + sour + H2O + salt + bitter,
    4: sour + H2O + sweet + H2O + salt + bitter,
    5: salt + H2O + sweet + H2O + sour + bitter,
    6: sweet + H2O + salt + H2O + sour + bitter,
    7: sour + H2O + salt + sweet + H2O + bitter,
    8: salt + H2O + sour + sweet + H2O + bitter,
    9: sweet + H2O + sour + salt + H2O + bitter,
    10: sour + H2O + sweet + salt + H2O + bitter,
    11: salt + H2O + sweet + sour + H2O + bitter,
    12: sweet + H2O + salt + sour + H2O + bitter,
    13: sour + salt + H2O + sweet + H2O + bitter,
    14: salt + sour + H2O + sweet + H2O + bitter,
    15: sweet + sour + H2O + salt + H2O + bitter,
    16: sour + sweet + H2O + salt + H2O + bitter,
    17: salt + sweet + H2O + sour + H2O + bitter,
    18: sweet + salt + H2O + sour + H2O + bitter
    }
  
  # Group column names
  group_colName = 'cohort_id'
