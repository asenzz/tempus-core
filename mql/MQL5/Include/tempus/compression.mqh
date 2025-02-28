#import "tempus-compression.dll"
   int LZ4_versionNumber();

   uint getMaxBufferSize();
   int  getCompressBound(int);

   int createCompressor();
   int compress(const uint compressor, short const& what[], char & dest[], const uint whatLen, const uint destLen);
   void deleteCompressor(uint compressor);

   int createDecompressor();
   /**************************************************************************/
   /*                     I M P O R T A N T                                  */
   /*                                                                        */
   /* The last uint32_t for decompress shoud be the uncopressed initial      */
   /* buffer size                                                            */
   /**************************************************************************/
   int decompress(uint decompressor, char const & what[], short const & dest[], uint whatLen, uint uncompressedLen);
   void deleteDecompressor(uint decompressor);

   void dumpBinaryData(string const & what, uint whatLen);
#import

class CompStream
{
   int compressor;
public:
   CompStream() { compressor = createCompressor(); if(compressor < 0) { int p[1]; p[2]=1; } }
   ~CompStream() { deleteCompressor(compressor); }
   int compress(string const & what, char & result[])
   {
      uint const whatLen = StringLen(what)*2, bufSz = getCompressBound(whatLen);
      short whatBuf[]; ArrayResize(whatBuf, whatLen / 2); StringToShortArray(what, whatBuf, 0, whatLen / 2 );
      
      ArrayResize(result, bufSz);
      int sz = compress(compressor, whatBuf, result, whatLen, bufSz);
      ArrayResize(result, sz);
      return sz;
   }
};

class DecompStream
{
   int decompressor;
public:
   DecompStream() { decompressor = createDecompressor(); if(decompressor < 0) { int p[1]; p[2]=1; } }
   ~DecompStream() { deleteDecompressor(decompressor); }
   
   int decompress(char const & what[], string & result, uint initialStringSize)
   {
      uint const whatLen = ArraySize(what);
      ushort buffer[]; ArrayResize(buffer, initialStringSize);
      int sz = decompress(decompressor, what, buffer, whatLen, initialStringSize * 2);
      result = ShortArrayToString(buffer, 0, sz);
      return sz;
   }
};

