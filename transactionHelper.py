import io
import sys
from sklearn.feature_extraction.text import CountVectorizer

class TransactionHelper:
    '''
    I use CountVectorizer as an easy way to keep track of frequency of transactions
    I changed token_pattern to keep all digit transactions
    I consider all n-grams from rang 3 to 10
    '''
    def __init__( self, dataLocation ):
        self.countVect = CountVectorizer( ngram_range = ( 3, 10 ), token_pattern = '\d+' )
        self.freqDict = self.invertDict( self.getTransactionFrq( dataLocation ) )

    def invertDict(self, mydict):
        '''
        this is needed to create <freq :[transaction list])
        just to make the look up easy and scalable
        '''
        inv_map = {}
        for k,v in mydict.iteritems():
            inv_map[v] = inv_map.get( v, [] )
            inv_map[v].append( k )
        return inv_map

    def getTransactionFrq(self, infile):
        '''
        this function basically performs the tokenizing and
        calculates the total frequency by summing each column of the matrix
        we note that this process is needed only once
        the out put is a dict <transaction : total_freq>
        I organized the code so we can analyze the transaction files once and then
        obtain the frequent transactions as many as we want based on the specified sigma
        (preferably from another class)
        '''
        with io.open( infile, 'r' ) as fin:
            mx = self.countVect.fit_transform( fin )
            vocabs = list( self.countVect.get_feature_names() )
            freq = mx.sum( axis = 0 ).A1
            return dict( zip( vocabs, freq ) )
    
    
    def getFreqTransaction( self, filePath, sigma = 4 ):
        results = self.freqDict[sigma]
        fout = open( filePath, 'w' )
        fout.write( str( len( results ) ) )
        fout.write( ',' )
        fout.write( str( sigma ) )
        fout.write( ',' )
        fout.write( ','.join( results ) )
        fout.close()
        


if __name__ == "__main__":

    fileLocation = sys.argv[1]
    sigma = int( sys.argv[2] )
    output_path = sys.argv[3]
    transHelper = TransactionHelper( fileLocation )
    transHelper.getFreqTransaction( output_path, sigma )
        
