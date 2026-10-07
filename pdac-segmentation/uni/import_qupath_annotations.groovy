/**
 * Import the tissue classifier predictions (GeoJSON from export_qupath_annotations.py) into a QuPath project.
 *
 * Open the project, open this script in Automate > Script editor and choose Run > Run for project. For every image,
 * e.g. "TM105_HE.vsi - 20x_BF_01", the script reads TM105_HE_prediction.geojson from geojsonDir, adds its annotations
 * and saves the image data (.qpdata). The imported annotations are named "UNI prediction"; running the script again
 * replaces them and leaves all other annotations untouched.
 */
import qupath.lib.io.PathIO

// Folder with the <sample>_prediction.geojson files; by default the folder "predictions" in the project folder.
def geojsonDir = buildFilePath(PROJECT_BASE_DIR, "predictions")
def predictionName = "UNI prediction"

def entry = getProjectEntry()
def sample = entry.getImageName().split("\\.vsi")[0]
def file = new File(geojsonDir, sample + "_prediction.geojson")
if (!file.exists()) {
    println("${entry.getImageName()}: no ${file.getName()} in ${geojsonDir}, skipped")
    return
}

removeObjects(getAnnotationObjects().findAll { it.getName() == predictionName }, true)
def annotations = PathIO.readObjects(file)
annotations.each { it.setName(predictionName) }
addObjects(annotations)
entry.saveImageData(getCurrentImageData())
println("${entry.getImageName()}: imported ${annotations.size()} annotations from ${file.getName()}")
