import processes.CleanCSV as CleanCSV
import processes.LabelMapping as LabelMapping
import processes.MongoExport as MongoExport

# Start all the id map extraction processes
MongoExport.main()
CleanCSV.main()
LabelMapping.main()