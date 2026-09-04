import qupath.lib.io.PathIO

if (args.length == 0 || args.length % 2 != 0) {
    throw new IllegalArgumentException(
        "Expected pairs of input .qpdata and output .geojson paths"
    )
}

for (int i = 0; i < args.length; i += 2) {
    def inputFile = new File(args[i])
    def outputFile = new File(args[i + 1])
    def hierarchy = PathIO.readHierarchy(inputFile)
    def annotations = hierarchy.getAnnotationObjects()
    PathIO.exportObjectsAsGeoJSON(outputFile, annotations)
    println("Exported ${annotations.size()} annotations from ${inputFile.name}")
}
