#include <iostream>
#include <string>
#include <fstream>
#include <dirent.h>
#include <map>
#include <set>
#include <math.h>



using namespace std;

// <word,counts of present in training files>
map<string, int> dictionary;

// <class, count of training files representing each class>
map<string, int> message_class_count;

// iterators
map<string, int>::iterator it;
map<string, int>::iterator it2;
map<string, double>::iterator it_dbl;
map<string, map<string, int> >::iterator it_sms;
map<string, map<string, double> >::iterator it_ssd;
int sumarray[4] = {0, 0,  0, 0};
int actarray[4] = {101,101,101,101};
string sendback[4] = {"baseball","graphic","politic","space"};
// <for all words in dictionary, <for all classes, count words present in messages in training class >>
map<string, map<string, int> > mc_word_count;

//  <for all words in dictionary,<filename, counts for words by test file>>
map<string, map<string, int> > test_word_count;

// <for all classes, probability of a class occuring giving training inputs>
map<string, double> prob_class;

// <for all test messages, <class, prob that message belongs to each class >>
map<string, map<string, double> > prob_mcw_gc;
// < same, but negated probability
map<string, map<string, double> > q_prob_mcw_gc;

// <for all words in dictionary, <for all classes, probability that word belongs to a class from training files>
map<string, map<string, double> > prob_cm_gm;

//map<string, map<string, double> > prob_mcw_pc;

// <class, <message, prob>>
map<string, map<string, double> > prob_test_mcw_gc;

double sum = 0.0;
ofstream classify("classification.txt");
ofstream class_sum("classification-summary.txt");
// Setting up multi-dimensional map for messages per class, that contain a dictionary word
// <word, <class, count>>
void instantiate_m_pc_pw()
{
    for(it=dictionary.begin(); it != dictionary.end(); it++)
    {
        for(it2=message_class_count.begin(); it2 != message_class_count.end(); it2++)
        {
            // outer map pairing outline
            mc_word_count.insert(make_pair(it->first, map<string,int>()));
            
            // inner map pairing outline, initialized to 0 counts
            pair<string, int> tempg("graphics", 0);
            pair<string, int> tempb("baseball", 0);
            pair<string, int> temps("space", 0);
            pair<string, int> tempm("politics", 0);
            mc_word_count[it->first].insert(tempg);
            mc_word_count[it->first].insert(tempb);
            mc_word_count[it->first].insert(temps);
            mc_word_count[it->first].insert(tempm);
        }
    }
}
void compute_prob_w_pc_gm()
{
    // consider every word in dictionary
    for(it=dictionary.begin(); it != dictionary.end(); it++)
    {
        // consider each message class
        for(it2=message_class_count.begin(), it_dbl=prob_class.begin(); (it2 != message_class_count.end() && it_dbl != prob_class.end()); it2++, it_dbl++)
        {
            // outer key using word
            //  <word, <class, prob>>
            prob_cm_gm.insert(make_pair(it->first, map<string,double>()));
            double result = log(prob_mcw_gc[it->first][it2->first]) + log(prob_class[it_dbl->first]);
            pair<string, double> temp(it2->first,result);
            prob_cm_gm[it->first].insert(temp);
            
//            cout << "word : " << it->first << ", class : " << it2->first << ", prob of cm | m :  " << prob_mcw_gc[it->first][it2->first]  << endl;
        }
    }
}
// <class, <message, prob>>
void instantiate_prob_test_mcw_gc(string message)
{
        for(it=message_class_count.begin(); it != message_class_count.end(); it++)
        {
            // outer map pairing outline
            prob_test_mcw_gc.insert(make_pair(it->first, map<string,double>()));
            
            // inner map pairing outline, initialized to 0 counts
            pair<string, int> temp(message, 0);
            prob_test_mcw_gc[it->first].insert(temp);
        }
}



// compute/store w/ multi-dimensional map the probability per class, that contain a dictionary word
// <word, <class, count>>
void compute_prob_mcw_gc()
{
    remove( "network.txt" );
  // Create and open a text file
  ofstream MyFile("network.txt");


    // consider every word in dictionary
    for(it=dictionary.begin(); it != dictionary.end(); it++)
    {
        int index = 0;
        // consider each message class
        for(it2=message_class_count.begin(); it2 != message_class_count.end(); it2++)
        {
            
            // outer map pairing outline
            prob_mcw_gc.insert(make_pair(it->first, map<string,double>()));
            q_prob_mcw_gc.insert(make_pair(it->first, map<string,double>()));

            // find individual prob (word mention in message per class / messages per class)
            it_sms = mc_word_count.find(it->first);
            
            // laplace smoothed (x+1) / (y+2)
            double num = (it_sms->second.find(it2->first)->second) + 1;
            double denom = it2->second + 2;
            double result = ((double)num)/denom;
            // inner map pairing outline, initialized to 0 counts
            pair<string, double> temp(it2->first,result);
            pair<string, double> tempq(it2->first,1-result);
            prob_mcw_gc[it->first].insert(temp);
            q_prob_mcw_gc[it->first].insert(tempq);
            // string sendback[4] = {"baseball","graphic","politic","space"};
            int id = 0;
            if(index == 0)
                id = 0;
            if(index == 1)
                id = 1;
            if(index == 2)
                id = 2;
            if(index == 3)
                id = 3;
            MyFile << it->first << "\t" << id << "\t" << prob_mcw_gc[it->first][it2->first] << endl;
            index++;
        }
    }
    // Close the file
    MyFile.close();
}
string predictClassByMessage(string message)
{
    double x[4] = {0,0,0,0};
    for(it=dictionary.begin(); it != dictionary.end(); it++)
    {
        int index = 0;
        // consider each message class
        for(it2=message_class_count.begin(); it2 != message_class_count.end(); it2++)
        {
            int count = 0;
            double y = prob_mcw_gc.at(it->first).at(it2->first);
            
            it_sms = test_word_count.find(it->first);
            if(it_sms != test_word_count.end())
            {
                try{
                    if(it_sms->second.count(message))
                    {
                        count = it_sms->second.find(message)->second;
                    }
                }catch(exception ex){}
            }
            x[index++] += log(y) * count;
        }
    }
//    cout << x[0] << "\t" << x[1] << "\t" << x[2] << "\t" << x[3] << endl;
    double best = x[0];
    int bestidx = 0;
    for(int i=1;i<4;i++)
    {
        if(x[i] > best){
            best = x[i];
            bestidx = i;
        }
    }
    switch(bestidx)
    {
        case(0):
            sumarray[0]++;
            return sendback[0];
        case(1):
            sumarray[1]++;
            return sendback[1];
        case(2):
            sumarray[2]++;
            return sendback[2];
        case(3):
            sumarray[3]++;
            return sendback[3];
    }
    return "";
}

// provided; strips punctuation from words.
string ProcessWord(string s)
{
    string t;
    
    // Remove punctuation.
    for (int i = 0; i < s.size(); i++)
        if (!ispunct(s[i]))
            t += s[i];

    // Convert to lower case.
    std::transform(t.begin(), t.end(), t.begin(), ::tolower);
    
    return t;
}

// reads input file from filename, and updates data structures according to
// class of message and stage of main
void ParseFile(string filename, string mes_class, string trajectory)
{
    // Open file.
    ifstream in;
    in.open(filename.c_str());
    if (!in.is_open())
    {
        cout<<"File not found: "<<filename<<endl;
        return;
    }
    
    // Find the end of the header.
    if(filename != "dictionary.txt"){
        string line;
        while (getline(in, line))
        {
            if (line == "")
                break;
        }
    }
    
    // Read the rest of the file word by word.
    string word;
    set<string> words_counted;
    while (in >> word)
    {
        // Strip the word of punctuation and convert to lower case.
        word = ProcessWord(word);
        
        if (word != "")
        {
            // initial read of dictionary into map
            if(filename == "dictionary.txt")
            {
                dictionary.insert(std::pair<string,int>(word,0));
            }
            
            // updating word counts from message files
            else if(trajectory ==  "fp")
            {
                it = dictionary.find(word);
                if (it != dictionary.end())
                {
                    it->second++;
                }
            }
            // update word counts (w/ multiples for) test files
            if(trajectory == "test")
            {
                if(!test_word_count.count(word))
                {
                    //  <word, <message, count>>
                    test_word_count.insert(make_pair(word, map<string,int>()));
                    pair<string, int> tempg(filename, 1);
                    test_word_count[word].insert(tempg);
                } else
                {
                    test_word_count[word][filename]++;
                }
            }
//             updating word counts from message files, per class
            else
            {
                it_sms = mc_word_count.find(word);
                if (it_sms != mc_word_count.end())
                {
                    it = it_sms->second.find(mes_class);
                    if(it != it_sms->second.end() && words_counted.count(word) == 0)
                    {
                        it->second++;
                        words_counted.insert(word);
                    }
                }
            }
        }
    }
}

// explores all files of a given directory and feeds each to parsefile()
void traverseDir(string mes_class, char* path, string trajectory)
{
    DIR *dh;
    struct dirent * contents;
    dh = opendir ( path );
      if ( !dh )
      {
        cout << "The given directory is not found";
          return;
      }

      // loop through training files & increment
      while ( ( contents = readdir ( dh ) ) != NULL )
      {
          string temp = contents->d_name;
          string temp2 = path;
          temp2.append(temp);
          ParseFile(temp2, mes_class, trajectory);
          if(trajectory == "fp")
              message_class_count.find(mes_class)->second++;
          if(trajectory == "test")
          {
              instantiate_prob_test_mcw_gc(temp2);
              // Create and open a text file
              
              string best = predictClassByMessage(temp2);
              classify <<  path << "\t" << best << "\t" << mes_class << endl;
              
          }
      }
    closedir ( dh );
    
}
int main()
{
    // build out dictionary mapping
    ParseFile("dictionary.txt", "","d");
    // instantiate map for messages per class counts
    message_class_count.insert(pair<string,int>("graphics",0));
    message_class_count.insert(pair<string,int>("baseball",0));
    message_class_count.insert(pair<string,int>("space",0));
    message_class_count.insert(pair<string,int>("politics",0));
    
    // instantiate map for messages per class probabilities
    prob_class.insert(pair<string,double>("graphics",0));
    prob_class.insert(pair<string,double>("baseball",0));
    prob_class.insert(pair<string,double>("space",0));
    prob_class.insert(pair<string,double>("politics",0));
    
    // pass paths to directories for ingest
    char * path_graphics = (char*)"training/comp.graphics/";
    char * path_baseball = (char*)"training/rec.sport.baseball/";
    char * path_space = (char*)"training/sci.space/";
    char * path_politics = (char*)"training/talk.politics.mideast/";
    traverseDir("graphics", path_graphics, "fp");
    traverseDir("baseball", path_baseball, "fp");
    traverseDir("space", path_space, "fp");
    traverseDir("politics", path_politics, "fp");
    
    // sum total messages
    for(it=message_class_count.begin(); it != message_class_count.end(); it++)
    {
        sum += it->second;
    }
    
    // calculate/update probabilities of messages based on class
    for(it_dbl=prob_class.begin(); it_dbl != prob_class.end(); it_dbl++)
    {
        it_dbl->second = ((double)message_class_count.find(it_dbl->first)->second / sum);
    }
	
    instantiate_m_pc_pw();
    
    // implement messages belonging to class c that contain w
    traverseDir("graphics", path_graphics, "sp");
    traverseDir("baseball", path_baseball, "sp");
    traverseDir("space", path_space, "sp");
    traverseDir("politics", path_politics, "sp");
    
    // calculate prob of message from class c contains word given class
    compute_prob_mcw_gc();
    
    // sum over all words [log(Pl(x|c)) + Log(P(c))]
    compute_prob_w_pc_gm();
    
    // injest test files
    ParseFile("test/comp.graphics/37926", "","test");
    char * path_tgraphics = (char*)"test/comp.graphics/";
    char * path_tbaseball = (char*)"test/rec.sport.baseball/";
    char * path_tspace = (char*)"test/sci.space/";
    char * path_tpolitics = (char*)"test/talk.politics.mideast/";
    traverseDir("graphics", path_tgraphics, "test");
    traverseDir("baseball", path_tbaseball, "test");
    traverseDir("space", path_tspace, "test");
    traverseDir("politics", path_tpolitics, "test");
    
    // <word, <message, count>>
    // use message word counts to determine class probablities
//    ParseFile("test/comp.graphics/37926", "graphics", "test");
//    instantiate_prob_test_mcw_gc("test/comp.graphics/37926");
//    predictClassByMessage("test/comp.graphics/37926");
    for(int i=0; i<4; i++)
    {
        class_sum << sendback[i];
        if(i>0)
        {
            class_sum <<"  ";
        }
        class_sum <<sumarray[i] << "\t" << actarray[i] << "\n";
    }
    classify.close();
    class_sum.close();
    return 0;
}
