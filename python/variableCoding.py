
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
      'Image VI' : 'cross'}

  # dictionary with index - image number correspondance
  imageDecoder = {0: 'Image I',
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

  orders_id = '_presOrder'
  pres_order_fileID = 'day1_presOrder'
  pres_order_d2_fileID = 'day2_presOrder'
  pres_order_d3_fileID = 'day3_presOrder'

  ratings_id = '_ratings_'
  flav_ratings_fileID = 'day1_rating'
  flav_ratings_d2_fileID = 'day2_rating'
  flav_ratings_d3_fileID = 'day3_rating'

  intensity_colName = 'intensity'
  novelty_colName = 'novelty'
  pleasanteness_colName = 'pleasantness'

  learn_order_fileID = 'day1_learnOrder'
  learningOrder_colName = 'Learning order'
  imageName_colName = 'image'
  imageID_colName = 'image_id'

  assocTestOrder1_colName = 'Day 1 - Association Test order'
  assoc1_fileID = 'day1_atest'
  assoc1_order_fileID = 'day1_atestOrder'

  assocTestOrder2_colName = 'Day 2 - Association Test order'
  assoc2_fileID = 'day2_atest'
  assoc2_order_fileID = 'day2_atestOrder'

  assocTestOrder3_colName = 'Day 3 - Association Test order'
  assoc3_fileID = 'day3_atest'
  assoc3_order_fileID = 'day3_atestOrder'

  neuroEconOrder0_fileID = 'day1_neuroEconOrder'
  # files containing the order of the trials in the neuroeconomics task performed at day 2 and 3
  neuroEconOrder1_fileID = 'day2_neuroEconOrder'
  neuroEconOrder2_fileID = 'day3_neuroEconOrder'
  
  conditioning_order_fileID = 'condOrder'
  conditioning_order_colName = 'Conditioning Order'

  # files containing the users' responses to the neuroeconomics task
  neuroEcon_id = '_neuroEcon_'
  neuroEcon_d1_responses_fileID = 'day1_neuroEcon'
  neuroEcon_d2_responses_fileID = 'day2_neuroEcon'
  neuroEcon_d3_responses_fileID = 'day3_neuroEcon'

  neurEconRealization_d1_fileID = 'day1_neurEconRealization'
  # trials that were actually realized after the task was completed
  neurEconRealization_d2_fileID = 'day2_neurEconRealization'
  neurEconRealization_d3_fileID = 'day3_neurEconRealization'
