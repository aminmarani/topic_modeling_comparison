import diskcache as dc

class db:
	def __init__(self,db_file):
		self.db_file = db_file
		self.db = dc.Cache(db_file)

		print('Load NPMI coherence DB. \nNumber of keys : {0}'.format(self.db.__len__()))


