import diskcache as dc

class db:
	def __init__(self,db_file):
		self.db_file = db_file
		self.db = dc.Cache(db_file)

		print('Load NPMI coherence DB. \nNumber of keys : {0}'.format(self.db.__len__()))

	def get(self,key):
		#try to return the value for the key
		try:
			return self.db[key]
		except KeyError:
			try:#if the key does not exist, try different combination of the key
				return self.db[(key[1],key[0])]
			except KeyError:#if no combinations of these pairs exist, return -100
				return -100

