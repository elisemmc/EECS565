//
// Author: Elise McEllhiney
//
// Created: Feb 10, 2016
//
// Modified: Feb 22, 2016
//
// Given cipher text, first word length, and key length this program will crack a vernier cipher
// The first word length must exceed the key length
// This algorith decreases in efficiency as the key length and the first word length become more similar
//

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
#include <vector>

#define ASCII_CAP_CONVERT 65

/*
//Problem 1:
#define CIPHER "MSOKKJCOSXOEEKDTOSLGFWCMCHSUSGX"
#define FIRST_WORD_LENGTH 6
#define KEY_LENGTH 2

//Problem 2:
#define CIPHER "OOPCULNWFRCFQAQJGPNARMEYUODYOUNRGWORQEPVARCEPBBSCEQYEARAJUYGWWYACYWBPRNEJBMDTEAEYCCFJNENSGWAQRTSJTGXNRQRMDGFEEPHSJRGFCFMACCB"
#define FIRST_WORD_LENGTH 7
#define KEY_LENGTH 3

//Problem 3:
#define CIPHER "MTZHZEOQKASVBDOWMWMKMNYIIHVWPEXJA"
#define FIRST_WORD_LENGTH 10
#define KEY_LENGTH 4

//Problem 4:
#define CIPHER "HUETNMIXVTMQWZTQMMZUNZXNSSBLNSJVSJQDLKR"
#define FIRST_WORD_LENGTH 11
#define KEY_LENGTH 5

//Problem 5:
#define CIPHER "LDWMEKPOPSWNOAVBIDHIPCEWAETYRVOAUPSINOVDIEDHCDSELHCCPVHRPOHZUSERSFS"
#define FIRST_WORD_LENGTH 9
#define KEY_LENGTH 6
*/

//Problem 6:
#define CIPHER "VVVLZWWPBWHZDKBTXLDCGOTGTGRWAQWZSDHEMXLBELUMO"
#define FIRST_WORD_LENGTH 13
#define KEY_LENGTH 7
/*

#define CIPHER "QSBWEBOQCXWKKVR"
#define FIRST_WORD_LENGTH 3
#define KEY_LENGTH 3
*/
/**************************************************************************
 * CUDA Functions
 **************************************************************************/
 __global__ void generateKeys( char* cipher, char* testWord, char* testKeys, char* keys, int firstWordLength, int keyLength )
 {
    int i;
    
    for( i = 0; i < firstWordLength; i++ )
    {
        testKeys[ ( blockIdx.x * blockDim.x + threadIdx.x ) * 15 + i ] = ( ( cipher[ i ] - ASCII_CAP_CONVERT ) - ( testWord[ ( blockIdx.x * blockDim.x + threadIdx.x ) * firstWordLength + i ] - ASCII_CAP_CONVERT ) + 26 ) % 26 + ASCII_CAP_CONVERT;
        if( i < keyLength )
        {
            keys[ ( blockIdx.x * blockDim.x + threadIdx.x ) * keyLength + i ] = testKeys[ ( blockIdx.x * blockDim.x + threadIdx.x ) * 15 + i ];
        }
    }
    for( i = 0; i < keyLength; i++ )
    {
        int j = 0;

        while( j + i + keyLength < firstWordLength )
        {
            if( testKeys[ ( blockIdx.x * blockDim.x + threadIdx.x ) * 15 + j + i ] != testKeys[ ( blockIdx.x * blockDim.x + threadIdx.x ) * 15 + j + i + keyLength ] )
            {
                keys[ ( blockIdx.x * blockDim.x + threadIdx.x ) * keyLength ] = 'a';
                break;
            }

            j = j + keyLength;
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

    Matrix_out.open("key.csv");

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
 * Check output
 * This is by far the slowest way to check, it is only called for viable keys
 * If the key length and first words length are the same length this function
 * will be called for all possible first words.
 **************************************************************************/
std::string parseString( std::string input, std::string dict )
{
    std::string output = "a";

    if( input.length() == 0 )
    {
        output = "";
    }

    for(int i = 15; i > 0; i--)
    {
        if( input.length() >= i )
        {
            if( dict.find( ( "," + input.substr(0,i) + "," ) ) != std::string::npos )
            {
                output = input.substr(0,i) + parseString( input.substr( i, input.length()-i ), dict );
            }

            if(output.back() != 'a')
            {
                    break;
            }
        }
    }

    return output;
}//end parseString

/**************************************************************************
 * Encode/Decode Input
 **************************************************************************/
std::string processCipher( std::string input, std::string key, bool encode )
{
    //take out spaces and change all letters to lowercase
    input.erase( remove_if(input.begin(), input.end(), isspace), input.end() );
    std::transform(input.begin(), input.end(), input.begin(), ::toupper);
    std::transform(key.begin(), key.end(), key.begin(), ::toupper);

    std::string output = input;

    for(int i=0; i<input.length(); i++)
    {
        int keyValue = (int)key[i%key.length()] - ASCII_CAP_CONVERT;
        int textValue = (int)input[i] - ASCII_CAP_CONVERT;

        if(encode)
            output[i] = (char)( ( ( textValue + keyValue ) % 26 ) + ASCII_CAP_CONVERT );
        else
            output[i] = (char)( ( ( textValue + ( 26 - keyValue ) ) % 26 ) + ASCII_CAP_CONVERT );//I add so I don't have to deal with absolute values
    }

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

/**************************************************************************
 * Allow manual input through terminal to encode or decode text with known key
 **************************************************************************/
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

/**************************************************************************
 * Initialize dictionary
 **************************************************************************/
void initializeDictionary( char* one_letter, char* two_letter, char* three_letter, char* four_letter, char* five_letter
                         , char* six_letter, char* seven_letter, char* eight_letter, char* nine_letter, char* ten_letter
                         , char* eleven_letter, char* twelve_letter, char* thirteen_letter, char* fourteen_letter, char* fifteen_letter
                         , std::string &dict )
{
    std::ifstream dictionaryFile;
    dictionaryFile.open("dictionary.txt");

    std::string buffer;
    std::cout<<"Dictionary Upload:"<<std::endl;

    //one letter words
    int wLength = 1;
    one_letter[0] = 'A'; one_letter[1] = 'I'; one_letter[2] = 'O';
    dict.append("A,"); dict.append("I,"); dict.append("O,");

    for(int i = 3; i < 1024; i++)
    {
        for(int j = 0; j < wLength; j++)
        {
            one_letter[wLength*i+j]=buffer[j];
        }
    }
    std::cout<<": 1 : ";

    //two letter words
    wLength = 2;
    for(int i = 0; i < 96; i++)
    {
        getline(dictionaryFile, buffer);
        for(int j = 0; j < wLength; j++)
        {
            two_letter[wLength*i+j]=buffer[j];
        }
        dict.append(buffer.substr(0,wLength)+",");
    }
    for(int i = 96; i < 1024; i++)
    {
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
        dict.append(buffer.substr(0,wLength)+",");
    }
    for(int i = 972; i < 1024; i++)
    {
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
        dict.append(buffer.substr(0,wLength)+",");
    }
    for(int i = 3903; i < 1024*4; i++)
    {
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
        dict.append(buffer.substr(0,wLength)+",");
    }
    for(int i = 8636; i < 1024*9; i++)
    {
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
        dict.append(buffer.substr(0,wLength)+",");
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
        dict.append(buffer.substr(0,wLength)+",");
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
        dict.append(buffer.substr(0,wLength)+",");
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
        dict.append(buffer.substr(0,wLength)+",");
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
        dict.append(buffer.substr(0,wLength)+",");
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
        dict.append(buffer.substr(0,wLength)+",");
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
        dict.append(buffer.substr(0,wLength)+",");
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
        dict.append(buffer.substr(0,wLength)+",");
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
        dict.append(buffer.substr(0,wLength)+",");
    }
    for(int i = 5059; i < 1024*5; i++)
    {
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
        dict.append(buffer.substr(0,wLength)+",");
    }
    for(int i = 3157; i < 1024*4; i++)
    {
        for(int j = 0; j < wLength; j++)
        {
            fifteen_letter[wLength*i+j]=buffer[j];
        }
    }
    std::cout<<"15 :\n";

    dictionaryFile.close();
}

/**************************************************************************
 * Main
 **************************************************************************/
int main(int argc, char **argv)
{
    //giveOption();

    struct timespec tstart, tend;

    clock_gettime(CLOCK_REALTIME, &tstart);

    //
    //initialize dictionary
    //
    char *one_letter, *two_letter, *three_letter, *four_letter, *five_letter
         , *six_letter, *seven_letter, *eight_letter, *nine_letter, *ten_letter
         , *eleven_letter, *twelve_letter, *thirteen_letter, *fourteen_letter, *fifteen_letter;

    std::string dictionaryStr = ",";

    //create arrays for all words of a given length in dictionary
    //for words I generate keys from I group them in batches of 1024, so I have filler text of all Z
    cudaMallocManaged((void **)&one_letter, sizeof(char)*1024*1);//3 one letter words
    cudaMallocManaged((void **)&two_letter, sizeof(char)*2*1024*1);//96 two letter words
    cudaMallocManaged((void **)&three_letter, sizeof(char)*3*1024*1);//972 three letter words
    cudaMallocManaged((void **)&four_letter, sizeof(char)*4*1024*4);//3903 four letter words
    cudaMallocManaged((void **)&five_letter, sizeof(char)*5*1024*9);//8636 five letter words
    cudaMallocManaged((void **)&six_letter, sizeof(char)*6*1024*15);//15232 six letter words
    cudaMallocManaged((void **)&seven_letter, sizeof(char)*7*1024*23);//23109 seven letter words
    cudaMallocManaged((void **)&eight_letter, sizeof(char)*8*1024*28);//28419 eight letter words
    cudaMallocManaged((void **)&nine_letter, sizeof(char)*9*1024*25);//24793 nine letter words
    cudaMallocManaged((void **)&ten_letter, sizeof(char)*10*1024*20);//20197 ten letter words
    cudaMallocManaged((void **)&eleven_letter, sizeof(char)*11*1024*16);//15407 eleven letter words
    cudaMallocManaged((void **)&twelve_letter, sizeof(char)*12*1024*11);//11248 twelve letter words
    cudaMallocManaged((void **)&thirteen_letter, sizeof(char)*13*1024*8);//7736 thirteen letter words
    cudaMallocManaged((void **)&fourteen_letter, sizeof(char)*14*1024*5);//5059 fourteen letter words
    cudaMallocManaged((void **)&fifteen_letter, sizeof(char)*15*1024*4);//3157 fifteen letter words

    initializeDictionary( one_letter, two_letter, three_letter, four_letter, five_letter
                        , six_letter, seven_letter, eight_letter, nine_letter, ten_letter
                        , eleven_letter, twelve_letter, thirteen_letter, fourteen_letter, fifteen_letter
                        , dictionaryStr );
    //
    //dictionary loaded into 1d character arrays
    //
    clock_gettime(CLOCK_REALTIME, &tend);
    printf("dictionary upload: %ld usec\n", get_elapsed(&tstart, &tend)/1000);

    int blocks, *key_length, *first_word_length;
    char *dictionaryArray, *input_chars, *test_key_holder, *keys;

    //
    //set dependant variables
    //
    switch( FIRST_WORD_LENGTH )
    {
        case 1 : blocks = 1; dictionaryArray = one_letter; break;
        case 2 : blocks = 1; dictionaryArray = two_letter; break;
        case 3 : blocks = 1; dictionaryArray = three_letter; break;
        case 4 : blocks = 4; dictionaryArray = four_letter; break;
        case 5 : blocks = 9; dictionaryArray = five_letter; break;
        case 6 : blocks = 15; dictionaryArray = six_letter; break;
        case 7 : blocks = 23; dictionaryArray = seven_letter; break;
        case 8 : blocks = 28; dictionaryArray = eight_letter; break;
        case 9 : blocks = 25; dictionaryArray = nine_letter; break;
        case 10 : blocks = 20; dictionaryArray = ten_letter; break;
        case 11 : blocks = 16; dictionaryArray = eleven_letter; break;
        case 12 : blocks = 11; dictionaryArray = twelve_letter; break;
        case 13 : blocks = 8; dictionaryArray = thirteen_letter; break;
        case 14 : blocks = 5; dictionaryArray = fourteen_letter; break;
        case 15 : blocks = 4; dictionaryArray = fifteen_letter; break;
    }
    //
    //variables set
    //

    cudaMallocManaged((void **)&test_key_holder, sizeof(char)*15*1024*blocks);//allocate for maximum possible length words so we can generate keys for any length word

    std::string inputString = CIPHER;
    cudaMallocManaged((void **)&input_chars, sizeof(char)*inputString.length());
    for(int i = 0; i < inputString.length(); i++ )
    {
        input_chars[i] = inputString[i];
    }

    cudaMallocManaged((void **)&keys, sizeof(char)*KEY_LENGTH*1024*blocks);

    /* Filter outputs */
    clock_gettime(CLOCK_REALTIME, &tstart);
    generateKeys<<< blocks, 1024 >>>( input_chars, dictionaryArray, test_key_holder, keys, FIRST_WORD_LENGTH, KEY_LENGTH);
    /* cuda synchronize */
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_REALTIME, &tend);
    printf("cuda key generation: %ld usec\n", get_elapsed(&tstart, &tend)/1000);

    clock_gettime(CLOCK_REALTIME, &tstart);
    for(int i = 0; i < 1024*blocks; i++)
    {
        std::string testKey;

        if(keys[KEY_LENGTH*i] != 'a')
        {
            for(int j = 0; j < KEY_LENGTH; j++)
            {
                testKey += keys[KEY_LENGTH*i+j];
            }
            
            std::string plainText = processCipher(CIPHER, testKey, false); 
            /*
            std::string testString = plainText.substr(0,FIRST_WORD_LENGTH) + parseString( plainText.substr( FIRST_WORD_LENGTH, plainText.length()-FIRST_WORD_LENGTH ), dictionaryStr );
            */

            std::string testString = parseString(plainText, dictionaryStr);
            if( testString.back() != 'a' )
            {
                std::cout << "Key: " << testKey << std::endl;
                std::cout << "Plain Text: " << testString << std::endl;
            }

        }
    }
    clock_gettime(CLOCK_REALTIME, &tend);
    printf("sentence check: %ld usec\n", get_elapsed(&tstart, &tend)/1000);

    cudaFree(input_chars);
    cudaFree(keys);

    cudaFree(test_key_holder);
    
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
  
    return 0;
}//end main
