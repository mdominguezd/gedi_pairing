import ee
import geemap

# Initialize Google Earth Engine
ee.Initialize()

# Define the Area of Interest (Spain)
spain = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filter(ee.Filter.eq('country_na', 'Spain'))
spain_geometry = spain.geometry()

# Define the time period
start_date = '2018-01-01'
end_date = '2020-12-31'

def add_ndvi(image):
    """Add NDVI to Sentinel-2 image"""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def add_vegetation_indices(image):
    """Add SAVI, gNDVI, and NDWI to Sentinel-2 image"""
    # NDVI: already added by add_ndvi
    # SAVI: (NIR - RED) / (NIR + RED + L) * (1 + L), L=0.5
    savi = image.expression(
        '((NIR - RED) / (NIR + RED + L)) * (1 + L)', {
            'NIR': image.select('B8'),
            'RED': image.select('B4'),
            'L': 0.5
        }
    ).rename('SAVI')
    # gNDVI: (NIR - GREEN) / (NIR + GREEN)
    gndvi = image.normalizedDifference(['B8', 'B3']).rename('gNDVI')
    # NDWI: (NIR - SWIR) / (NIR + SWIR)
    ndwi = image.normalizedDifference(['B8', 'B11']).rename('NDWI')
    return image.addBands([savi, gndvi, ndwi])

def mask_clouds_s2(image):
    """Mask clouds in Sentinel-2 images using the MSK_CLDPRB band"""
    cloud_prob = image.select('MSK_CLDPRB')
    # Mask pixels with cloud probability > 50%
    cloud_mask = cloud_prob.lt(50)
    return image.updateMask(cloud_mask).divide(10000)

def create_seasonal_composites(aoi=spain_geometry):
    """Create seasonal NDVI composites"""

    # Load Sentinel-2 collection
    s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(aoi) \
        .filterDate('2019-01-01', '2021-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) 

    # Summer composite (June-September)
    summer_collection = s2_collection.filter(ee.Filter.calendarRange(6, 8, 'month'))
    summer_ndvi = summer_collection.map(mask_clouds_s2).map(add_ndvi).select('NDVI').median().rename('NDVI_summer')
    summer_vi = summer_collection.map(mask_clouds_s2).map(add_vegetation_indices).select(['SAVI', 'gNDVI', 'NDWI']).median().rename(['SAVI_summer', 'gNDVI_summer', 'NDWI_summer'])

    # Winter composite (December-March)
    winter_collection = s2_collection.filter(
        ee.Filter.Or(
            ee.Filter.calendarRange(12, 12, 'month'),  # December
            ee.Filter.calendarRange(1, 3, 'month')     # January to March
        )
    )
    winter_ndvi = winter_collection.map(mask_clouds_s2).map(add_ndvi).select('NDVI').median().rename('NDVI_winter')
    winter_vi = winter_collection.map(mask_clouds_s2).map(add_vegetation_indices).select(['SAVI', 'gNDVI', 'NDWI']).median().rename(['SAVI_winter', 'gNDVI_winter', 'NDWI_winter'])

    
    # Combine seasonal composites
    return summer_ndvi.addBands(winter_ndvi).addBands(summer_vi).addBands(winter_vi)

def prepare_training_data():
    """Prepare training data from Copernicus landcover"""
    
    # Load Copernicus landcover data
    landcover = ee.Image('COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019')
    forest_type = landcover.select('forest_type')
    
    # Create mask to exclude forest type 0 (unknown) and non-forest areas
    forest_mask = forest_type.gt(0).And(forest_type.lte(5))
    
    # Apply mask
    forest_type_masked = forest_type.updateMask(forest_mask)
    
    # Resample to 10m to match Sentinel-2 resolution
    forest_type_10m = forest_type_masked.resample('bilinear').reproject(
        crs='EPSG:4326',
        scale=10
    )
    
    return forest_type_10m

def collect_training_samples(covariates, labels, n_samples=5000):
    """Collect training samples"""
    
    # Combine covariates and labels
    training_image = covariates.addBands(labels.rename('forest_type'))
    
    # Generate random points for sampling
    points = ee.FeatureCollection.randomPoints(
        region=spain_geometry,
        points=n_samples,
        seed=42
    )
    
    # Sample the image at random points
    training_data = training_image.sampleRegions(
        collection=points,
        properties=[],
        scale=10,
        projection='EPSG:4326',
        tileScale=4,
        geometries=True
    )
    
    return training_data

def train_random_forest(training_data):
    """Train Random Forest classifier"""
    
    # Define covariate bands
    covariate_bands = ['NDVI_summer', 'NDVI_winter', 'SAVI_winter', 'gNDVI_winter', 'NDWI_winter', 'SAVI_summer', 'gNDVI_summer', 'NDWI_summer']
    
    # Train the classifier
    rf_classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=20,
        variablesPerSplit=None,
        minLeafPopulation=1,
        bagFraction=0.5,
        maxNodes=10000,
        seed=42
    )
    
    trained_classifier = rf_classifier.train(
        features=training_data,
        classProperty='forest_type',
        inputProperties=covariate_bands
    )
    
    return trained_classifier, covariate_bands

def classify_forests(covariates, classifier):
    """Apply the trained classifier to create forest type map"""
    
    # Classify the image
    classified = covariates.classify(classifier)
    
    return classified.rename('forest_type_predicted')

def export_results(classified_image, description='spain_forest_classification_simple'):
    """Export classification results"""
    
    # Export to Google Drive
    export_task = ee.batch.Export.image.toDrive(
        image=classified_image.uint8(),
        description=description,
        folder='GEE_Exports',
        region=spain_geometry,
        scale=10,
        maxPixels=1e13,
        crs='EPSG:4326'
    )
    
    export_task.start()
    print(f"Export task started: {description}")
    return export_task

def visualize_results(classified_image):
    """Visualize classification results"""
    
    # Define visualization parameters
    forest_vis = {
        'min': 1,
        'max': 4,
        'palette': ['#1f4e23', '#8bc34a', '#2e7d32', '#66bb6a']  # Different shades of green
    }
    
    # Create map
    Map = geemap.Map(center=[40.0, -4.0], zoom=6)
    Map.addLayer(classified_image, forest_vis, 'Forest Types')
    Map.addLayer(spain.style(fillColor='00000000', color='red', width=2), {}, 'Spain Boundary')
    
    # Add legend
    legend_labels = [
        'Broadleaved evergreen',
        'Broadleaved deciduous', 
        'Needleleaved evergreen',
        'Needleleaved deciduous'
    ]
    
    Map.add_legend(title="Forest Types", labels=legend_labels, colors=forest_vis['palette'])
    
    return Map

# Main workflow
def main():
    """Main classification workflow"""
    
    print("Starting simplified forest type classification for Spain...")
    
    # Step 1: Create seasonal NDVI composites
    print("Creating seasonal NDVI composites...")
    ndvi_composites = create_seasonal_composites()
    
    # Step 2: Prepare target labels
    print("Preparing landcover labels...")
    forest_labels = prepare_training_data()
    
    # Step 3: Collect training samples
    print("Collecting training samples...")
    training_data = collect_training_samples(ndvi_composites, forest_labels, n_samples=5000)
    
    # Step 4: Train Random Forest classifier
    print("Training Random Forest classifier...")
    classifier, covariate_bands = train_random_forest(training_data)
    
    # Step 5: Apply classifier to create forest type map
    print("Applying classifier...")
    classified_forests = classify_forests(ndvi_composites, classifier)
    
    # Step 6: Export results
    print("Exporting results...")
    export_task = export_results(classified_forests, 'spain_forest_ndvi_classification')
    
    # Print information
    print("\nCovariate bands used:")
    print(covariate_bands)
    
    print("\nForest type classes:")
    print("1: Broadleaved evergreen")
    print("2: Broadleaved deciduous") 
    print("3: Needleleaved evergreen")
    print("4: Needleleaved deciduous")
    
    return classified_forests, classifier, training_data

# Run the classification
if __name__ == "__main__":
    classified_forests, classifier, training_data = main()
    
    # Create visualization
    Map = visualize_results(classified_forests)
    Map