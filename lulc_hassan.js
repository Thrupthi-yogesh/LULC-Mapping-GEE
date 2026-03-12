
//Add the Hassan district boundary to map 
Map.addLayer(hassan, {}, 'Hassan District Boundary');
Map.centerObject(hassan, 10);

//Load Landsat 9 SR collection and filtering
var L1 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2");
var Image = L1.filterBounds(hassan)
              .filterDate("2022-01-01","2022-12-01")
              .filterMetadata("CLOUD_COVER","less_than",5)
              .mean()
              .clip(hassan);
print(Image);
Map.addLayer(Image);

//Merge all 5 class training samples into one collection
var Training = vegetation.merge(water).merge(cropland).merge(barren_land).merge(buildup);
print(Training);

//Define 7 Landsat 9 spectral bands
var bands = ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'];

//Select the defined bands from the image for use in training and classification
var input = Image.select(bands);
var trainImage = Image.select(bands).sampleRegions({
  collection: Training,
  properties: ['Class'],
  scale: 60
});

//Add random column and split samples — 80% for training, 20% for testing
var trainData = trainImage.randomColumn();
var trainset = trainData.filter(ee.Filter.gte('random',0.8));
var testset  = trainData.filter(ee.Filter.lte('random',0.2));

// Train the SVM classifier using training samples
var classifier = ee.Classifier.libsvm().train({
  features: trainset,
  classProperty: 'Class',
  inputProperties: bands
});

//Apply trained SVM classifier to the image to produce the LULC classification map
var classified = Image.classify(classifier);
print(classified);

// Display classified LULC map with color palette
Map.addLayer(classified, {min:0, max:4, palette:['darkgreen','blue','lightgreen','red','yellow']}, 'Classified LULC');

//Export the classified LULC image
Export.image.toDrive({
  image: classified,
  description: 'Classified_Image',
  scale: 30,
  region: hassan,
  fileFormat: 'GeoTIFF'
});

// Step 12: Classify test samples using trained model and evaluate accuracy
var classified_Valid = testset.classify(classifier);
var confusionMatrix = classified_Valid.errorMatrix('Class','classification');
print('Confusion Matrix:', confusionMatrix);
print('Overall Accuracy:', confusionMatrix.accuracy());
print('Kappa Coefficient:', confusionMatrix.kappa());
