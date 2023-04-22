import gcsfs
key = ''
gcs_project = ''
root_dir = 'gs://bucket/data'

fs = gcsfs.GCSFileSystem(project=gcs_project,token=key)

a = fs.ls(root_dir)
print(a)