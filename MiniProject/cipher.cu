#include <algorithm>
#include <cctype>
#include <cuda_runtime.h>
#include <fstream>
#include <inttypes.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ASCII_CAP_CONVERT 65


#define CIPHER "MSOKKJCOSXOEEKDTOSLGFWCMCHSUSGX"
#define FIRST_WORD_LENGTH 6
#define KEY_LENGTH 2


/**************************************************************************
 * CUDA Functions
 **************************************************************************/
 __global__ void generateKeys( char* cipher, char* testWord, char* testKeys, char* keys )
 {
    int index = ( blockIdx.x * blockDim.x + threadIdx.x ) * FIRST_WORD_LENGTH;
    int keyIndex = ( blockIdx.x * blockDim.x + threadIdx.x ) * KEY_LENGTH;

    for(int i = 0; i < FIRST_WORD_LENGTH; i++)
    {
        testKeys[ index + i ] = ( ( cipher[ i ] - ASCII_CAP_CONVERT ) - ( testWord[ index + i ] - ASCII_CAP_CONVERT ) + 26 ) % 26 + ASCII_CAP_CONVERT;
        if( i < KEY_LENGTH )
        {
            keys[ keyIndex + i ] = testKeys[ index + i ];
        }
    }
    for( int i = 0; i < KEY_LENGTH; i++)
    {
        int j = 0;

        while( j + i + KEY_LENGTH < FIRST_WORD_LENGTH )
        {
            if( testKeys[ index + j + i ] != testKeys[ index + j + i + KEY_LENGTH ] )
            {
                keys[ keyIndex ] = 'a';
            }

            j = j + KEY_LENGTH;
        }
    }
 }

/**************************************************************************
 * Calculate time elapsed
 **************************************************************************/
unsigned long get_elapsed(struct timespec *start, struct timespec *end)
{
    uint64_t dur;
    dur = ((uint64_t)end->tv_sec * 1000000000 + end->tv_nsec) - 
        ((uint64_t)start->tv_sec * 1000000000 + start->tv_nsec);
    return (unsigned long)dur;
}//end get_elapsed

/**************************************************************************
 * Output to file
 **************************************************************************/
void file_output(char *data, int size, int word_length)
{
    std::ofstream Matrix_out;

    Matrix_out.open("out.csv");

    for(int i = 0; i < size; i++)
    {
        if(data[word_length*i] != 'a')
        {
            for(int j = 0; j < word_length; j++)
            {
                Matrix_out<<data[word_length*i+j];
            }
            Matrix_out<<",\n";
        }
    }

    Matrix_out.close();
}

/**************************************************************************
 * Encode/Decode Input
 **************************************************************************/
std::string processCipher( std::string input, std::string key, bool encode )
{
    struct timespec tstart, tend;

    clock_gettime(CLOCK_REALTIME, &tstart);

    //take out spaces and change all letters to lowercase
    input.erase( remove_if(input.begin(), input.end(), isspace), input.end() );
    std::transform(input.begin(), input.end(), input.begin(), ::toupper);
    std::transform(key.begin(), key.end(), key.begin(), ::toupper);

    std::string output = input;

    for(int i=0; i<input.length(); i++)
    {
        int keyValue = (int)key[i%key.length()] - 65;
        int textValue = (int)input[i] - 65;

        if(encode)
            output[i] = (char)( ( ( textValue + keyValue ) % 26 ) + 65 );
        else
            output[i] = (char)( ( ( textValue + ( 26 - keyValue ) ) % 26 ) + 65 );//I add so I don't have to deal with absolute values
    }

    clock_gettime(CLOCK_REALTIME, &tend);

    printf("Cipher Processing: %ld usec\n", get_elapsed(&tstart, &tend)/1000);

    return output;
}//end processCipher

/**************************************************************************
 * Run cipher with manual input and key values
 **************************************************************************/
void runCipher(bool encode)
{
    std::string input, output;
    std::string key;

    if(encode)
        std::cout << "Please input the text you wish to encode:" << std::endl;
    else
        std::cout << "Please input the text you wish to decode:" << std::endl;

    std::getline( std::cin, input );

    std::cout << "Please input the key:" << std::endl;
    std::getline( std::cin, key );

    std::cout << "Your cipher text is:" << std::endl;

    output = processCipher(input, key, encode);

    std::cout << output << std::endl;
}//end runCipher

void giveOption()
{
    std::string choice = "encode";
    bool loop = true;

    std::cout<<"Would you like to encode or decode?"<<std::endl;
    std::getline( std::cin, choice );

    while(loop)
    {
        if( choice == "encode")
        {
            runCipher(true);
            loop = false;
        }
        else if( choice == "decode")
        {
            runCipher(false);
            loop = false;
        }
        else
        {
            std::cout << "That is not a valid input, please input either 'encode' or 'decode'" << std::endl;
            std::getline( std::cin, choice );
        }
    }
}

int main(int argc, char **argv)
{
    //giveOption();

    struct timespec tstart, tend;

    char *one_letter, *two_letter, *three_letter, *four_letter, *five_letter
         , *six_letter, *seven_letter, *eight_letter, *nine_letter, *ten_letter
         , *eleven_letter, *twelve_letter, *thirteen_letter, *fourteen_letter, *fifteen_letter;

    //
    //initialize dictionary
    //
    std::ifstream dictionaryFile;
    dictionaryFile.open("dictionary.txt");

    //create arrays for all words of a given length in dictionary
    //for words I generate keys from I group them in batches of 1024, so I have filler text of all Z
    cudaMallocManaged((void **)&one_letter, sizeof(char)*3);//3 one letter words
    cudaMallocManaged((void **)&two_letter, sizeof(char)*2*96);//96 two letter words
    cudaMallocManaged((void **)&three_letter, sizeof(char)*3*972);//972 three letter words
    cudaMallocManaged((void **)&four_letter, sizeof(char)*4*3903);//3903 four letter words
    cudaMallocManaged((void **)&five_letter, sizeof(char)*5*8636);//8636 five letter words
    cudaMallocManaged((void **)&six_letter, sizeof(char)*6*1024*15);//15232 six letter words
    cudaMallocManaged((void **)&seven_letter, sizeof(char)*7*1024*23);//23109 seven letter words
    cudaMallocManaged((void **)&eight_letter, sizeof(char)*8*1024*28);//28419 eight letter words
    cudaMallocManaged((void **)&nine_letter, sizeof(char)*9*1024*25);//24793 nine letter words
    cudaMallocManaged((void **)&ten_letter, sizeof(char)*10*1024*20);//20197 ten letter words
    cudaMallocManaged((void **)&eleven_letter, sizeof(char)*11*1024*16);//15407 eleven letter words
    cudaMallocManaged((void **)&twelve_letter, sizeof(char)*12*1024*11);//11248 twelve letter words
    cudaMallocManaged((void **)&thirteen_letter, sizeof(char)*13*1024*8);//7736 thirteen letter words
    cudaMallocManaged((void **)&fourteen_letter, sizeof(char)*14*5059);//5059 fourteen letter words
    cudaMallocManaged((void **)&fifteen_letter, sizeof(char)*15*3157);//3157 fifteen letter words

    std::string buffer;

    //one letter words
    one_letter[0] = 'A'; one_letter[1] = 'I'; one_letter[2] = 'O';
    std::cout<<"Dictionary Upload:"<<std::endl;
    std::cout<<": 1 : ";
    //two letter words
    int wLength = 2;
    for(int i = 0; i < 96; i++)
    {
        getline(dictionaryFile, buffer);
        for(int j = 0; j < wLength; j++)
        {
            two_letter[wLength*i+j]=buffer[j];
        }
    }
    std::cout<<"2 : ";
    //three letter words
    wLength = 3;
    for(int i = 0; i < 972; i++)
    {
        getline(dictionaryFile, buffer);
        for(int j = 0; j < wLength; j++)
        {
            three_letter[wLength*i+j]=buffer[j];
        }
    }
    std::cout<<"3 : ";
    //four letter words
    wLength = 4;
    for(int i = 0; i < 3903; i++)
    {
        getline(dictionaryFile, buffer);
        for(int j = 0; j < wLength; j++)
        {
            four_letter[wLength*i+j]=buffer[j];
        }
    }
    std::cout<<"4 : ";
    //five letter words
    wLength = 5;
    for(int i = 0; i < 8636; i++)
    {
        getline(dictionaryFile, buffer);
        for(int j = 0; j < wLength; j++)
        {
            five_letter[wLength*i+j]=buffer[j];
        }
    }
    std::cout<<"5 : ";
    //six letter words
    wLength = 6;
    for(int i = 0; i < 15232; i++)
    {
        getline(dictionaryFile, buffer);
        for(int j = 0; j < wLength; j++)
        {
            six_letter[wLength*i+j]=buffer[j];
        }
    }
    for(int i = 15232; i < 1024*15; i++)
    {
        for(int j = 0; j < wLength; j++)
        {
            six_letter[wLength*i+j]='Z';//filler
        }
    }
    std::cout<<"6 : ";
    //seven letter words
    wLength = 7;
    for(int i = 0; i < 23109; i++)
    {
        getline(dictionaryFile, buffer);
        for(int j = 0; j < wLength; j++)
        {
            seven_letter[wLength*i+j]=buffer[j];
        }
    }
    for(int i = 23109; i < 1024*23; i++)
    {
        for(int j = 0; j < wLength; j++)
        {
            seven_letter[wLength*i+j]='Z';//filler
        }
    }
    std::cout<<"7 : ";
    //eight letter words
    wLength = 8;
    for(int i = 0; i < 28419; i++)
    {
        getline(dictionaryFile, buffer);
        for(int j = 0; j < wLength; j++)
        {
            eight_letter[wLength*i+j]=buffer[j];
        }
    }
    for(int i = 28419; i < 1024*28; i++)
    {
        for(int j = 0; j < wLength; j++)
        {
            eight_letter[wLength*i+j]='Z';
        }
    }
    std::cout<<"8 : ";
    //nine letter words
    wLength = 9;
    for(int i = 0; i < 24793; i++)
    {
        getline(dictionaryFile, buffer);
        for(int j = 0; j < wLength; j++)
        {
            nine_letter[wLength*i+j]=buffer[j];
        }
    }
    for(int i = 24793; i < 1024*25; i++)
    {
        for(int j = 0; j < wLength; j++)
        {
            nine_letter[wLength*i+j]='Z';
        }
    }
    std::cout<<"9 : ";
    //ten letter words
    wLength = 10;
    for(int i = 0; i < 20197; i++)
    {
        getline(dictionaryFile, buffer);
        for(int j = 0; j < wLength; j++)
        {
            ten_letter[wLength*i+j]=buffer[j];
        }
    }
    for(int i = 20197; i < 1024*20; i++)
    {
        for(int j = 0; j < wLength; j++)
        {
            ten_letter[wLength*i+j]='Z';
        }
    }
    std::cout<<"10 : ";
    //eleven letter words
    wLength = 11;
    for(int i = 0; i < 15407; i++)
    {
        getline(dictionaryFile, buffer);
        for(int j = 0; j < wLength; j++)
        {
            eleven_letter[wLength*i+j]=buffer[j];
        }
    }
    for(int i = 15407; i < 1024*16; i++)
    {
        for(int j = 0; j < wLength; j++)
        {
            eleven_letter[wLength*i+j]='Z';
        }
    }
    std::cout<<"11 : ";
    //twelve letter words
    wLength = 12;
    for(int i = 0; i < 11248; i++)
    {
        getline(dictionaryFile, buffer);
        for(int j = 0; j < wLength; j++)
        {
            twelve_letter[wLength*i+j]=buffer[j];
        }
    }
    for(int i = 11248; i < 1024*11; i++)
    {
        for(int j = 0; j < wLength; j++)
        {
            twelve_letter[wLength*i+j]='Z';
        }
    }
    std::cout<<"12 : ";
    //thirteen letter words
    wLength = 13;
    for(int i = 0; i < 7736; i++)
    {
        getline(dictionaryFile, buffer);
        for(int j = 0; j < wLength; j++)
        {
            thirteen_letter[wLength*i+j]=buffer[j];
        }
    }
    for(int i = 7736; i < 1024*8; i++)
    {
        for(int j = 0; j < wLength; j++)
        {
            thirteen_letter[wLength*i+j]='Z';
        }
    }
    std::cout<<"13 : ";
    //fourteen letter words
    wLength = 14;
    for(int i = 0; i < 5059; i++)
    {
        getline(dictionaryFile, buffer);
        for(int j = 0; j < wLength; j++)
        {
            fourteen_letter[wLength*i+j]=buffer[j];
        }
    }
    std::cout<<"14 : ";
    //fifteen letter words
    wLength = 15;
    for(int i = 0; i < 3157; i++)
    {
        getline(dictionaryFile, buffer);
        for(int j = 0; j < wLength; j++)
        {
            fifteen_letter[wLength*i+j]=buffer[j];
        }
    }
    std::cout<<"15 :\n";
    dictionaryFile.close();
    //
    //dictionary loaded into 1d character arrays
    //

    //
    //initialize dependant variables
    //
    int blocks;
    char* dictionaryArray;

    switch( FIRST_WORD_LENGTH )
    {
        case 6 : blocks = 15; dictionaryArray = six_letter; break;
        case 7 : blocks = 23; dictionaryArray = seven_letter; break;
        case 8 : blocks = 28; dictionaryArray = eight_letter; break;
        case 9 : blocks = 25; dictionaryArray = nine_letter; break;
        case 10 : blocks = 20; dictionaryArray = ten_letter; break;
        case 11 : blocks = 16; dictionaryArray = eleven_letter; break;
        case 12 : blocks = 11; dictionaryArray = twelve_letter; break;
        case 13 : blocks = 8; dictionaryArray = thirteen_letter; break;
    }
    //
    //variables initialized
    //

    int *key_length, *first_word_length;
    char *input_chars, *test_key_holder, *keys;

    std::string inputString = CIPHER;
    cudaMallocManaged((void **)&input_chars, sizeof(char)*inputString.length());
    for(int i = 0; i < inputString.length(); i++ )
    {
        input_chars[i] = inputString[i];
    }

    cudaMallocManaged((void **)&test_key_holder, sizeof(char)*FIRST_WORD_LENGTH*1024*blocks);
    cudaMallocManaged((void **)&keys, sizeof(char)*KEY_LENGTH*1024*blocks);

    /* Filter outputs */
    clock_gettime(CLOCK_REALTIME, &tstart);
    generateKeys<<< blocks, 1024 >>>( input_chars, dictionaryArray, test_key_holder, keys);
    /* cuda synchronize */
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_REALTIME, &tend);
    printf("cuda key generation: %ld usec\n", get_elapsed(&tstart, &tend)/1000);
    
    file_output(keys, 1024*blocks, KEY_LENGTH );

    cudaFree(one_letter);
    cudaFree(two_letter);
    cudaFree(three_letter);
    cudaFree(four_letter);
    cudaFree(five_letter);
    cudaFree(six_letter);
    cudaFree(seven_letter);
    cudaFree(eight_letter);
    cudaFree(nine_letter);
    cudaFree(ten_letter);
    cudaFree(eleven_letter);
    cudaFree(twelve_letter);
    cudaFree(thirteen_letter);
    cudaFree(fourteen_letter);
    cudaFree(fifteen_letter);

    cudaFree(input_chars);
  
    return 0;
}//end main
