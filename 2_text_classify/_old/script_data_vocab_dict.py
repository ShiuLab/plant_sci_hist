'''
For getting the library entries
07/02 Created
'''

import os, sys, getopt
import urllib3
from pathlib import Path
from time import sleep
from numpy.random import normal

def get_htmls(url_base, pg_base, en_base, out_name):
    """ Send url request
    Parameters:
     url_base - the base url
     pg_base  - the doi bit of the url 
     en_base  - the entry page extension to the url, a list
     s_time   - a numpy.array with sleep time, same size as en_base
    """
    
    out_path = Path(f"./{out_name}")
    
    # set sleep time
    s_time = normal(2, 0.5, len(en_base))
    
    try:
        os.mkdir(out_path)
    except:  # In case it is already created
        print("Folder exist:",out_name)
        sys.exit(0)
    
    for idx, en_list in enumerate(en_base):
        url = url_base + pg_base + en_list
        print("list",idx)
        response = http.request('GET', url)
        html = response.data.decode("utf-8")

        # Write the html for debugging purpose
        out_file = out_path/f"p_{idx}.html"
        
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(html)
                
        # Sleep
        st = s_time[idx]
        print (" sleep:",st)
        sleep(st)

def strip_tags(astr):
    ''' Strip off the outer most <...> and </...>'''
    astr = astr[astr.find(">")+1:astr.rfind("</")]
    return astr

def parse_htmls(out_name):
    ''' Parse all html files in the folder specified by out_name
    
    Target line example:
      <h2 class="itemTitle">
      <a href="/view/10.1093/acref/9780198833338.001.0001/acref-9780198833338-e-50?rskey=dM2QXw&amp;result=61">acid soil</a>
      </h2>
    '''
    
    print("\nGoing through html files:")
    htmls = os.listdir(out_name)
    
    # Note: for all IO, the encoding needs to be specified. Otherwise multiple
    #  UnicodeDecodeError and UnicodeEncodeError occurs
    oup = open(out_name+".items","w", encoding="utf-8")
    
    # Go through reach html file
    for html in htmls:
        #print(html)
        out_path = Path(f"./{out_name}")
        html_f   = out_path/html
        
        lines    = open(html_f, encoding="utf-8").readlines()
        
        found = 0
        items = []
        for i in range(len(lines)):
            if lines[i].find("itemTitle") != -1:
                found += 1
                # Put all lines between <a> and </a> into one string
                itemStr = lines[i+1].strip()
                # If this line does not have the end tag
                if itemStr.find("</a>") == -1:
                    for j in range(i+2, len(lines)):
                        itemStr += lines[j].strip()
                        # Found end tag
                        if itemStr.find("</a>") != -1:
                            break
                item = strip_tags(itemStr)
                
                # Odd situations
                #  <i>Acorus</i>
                #  Achira (<i>Canna edulis</i>)
                #  canopy((ecol.))
                if item.find("(") != -1:
                    if len(item.split("(")) > 2:
                        if item.find("((in") != -1:
                            item = item[:item.find("((in")]
                            items.append(item)
                        else:
                            print(f" {html} oddcase: {item}")
                            #item = item.split("(")[0]
                            #items.append(item)
                    else:
                        [item1, item2] = item.split("(")
                        items.append(item1.strip())
                        item2 = strip_tags(item2)
                        items.append(item2)
                elif item.find("<i>") != -1:
                    item = strip_tags(item)
                    items.append(item)
                else:
                    items.append(item)

        #print(f" found={found}, items={len(items)}")
        oup.write("\n".join(items) + "\n")
        # Oops, need to add an ending newline
        
    oup.close()

def get_topic_htmls(topic_param, out_name):
    
    (en_pg_num, pg_base) = topic_param 

    # Entry list links
    en_tag_L  = "?btog=chap&hide=true&page="
    en_tag_R  = "&pageSize=20"
    en_base   = [f"{en_tag_L}{i}{en_tag_R}" for i in range(1,en_pg_num)]    

    # Get htmls for the links
    get_htmls(url_base, pg_base, en_base, out_name)

def get_topic_param(topic):
    # dict:(num_html_pages, URL_part)
    pg = {
      "pln":(367 ,"10.1093/acref/9780199766444.001.0001/acref-9780199766444"),
      "bio":(294 ,"10.1093/acref/9780198821489.001.0001/acref-9780198821489"),
      "bmb":(1050,"10.1093/acref/9780198529170.001.0001/acref-9780198529170"),
      "evo":(21  ,"10.1093/acref/9780195122008.001.0001/acref-9780195122008"),
      "eco":(339 ,"10.1093/acref/9780191793158.001.0001/acref-9780191793158"),
      "gen":(367 ,"10.1093/acref/9780199766444.001.0001/acref-9780199766444"),
      "mth":(154 ,"10.1093/acref/9780199235940.001.0001/acref-9780199235940"),
      "sta":(118 ,"10.1093/acref/9780199679188.001.0001/acref-9780199679188"),
      "com":(326 ,"10.1093/acref/9780199688975.001.0001/acref-9780199688975")
    }
    return pg[topic]
    
def help():
    print('\nscript_data_vocab_dict.py -f <function> -o <out_name> -t <topic>')
    print('  -f functions:')
    print('      get_topic_htmls - for specific topic, save htmls to a folder')
    print('      parse_htmls - get item names out from htmls in a folder')
    print('  -o out_name: output base name, e.g. vocab_plant')
    print('  -t topics: pln, bmb, gen, bio, evo, eco')
    print('\n\n')

################################################################################

if __name__ == "__main__":
    user_agent = {'user-agent': 'Mozilla/5.0 (Windows NT 6.3; rv:36.0) ..'}
    http       = urllib3.PoolManager(10, headers=user_agent)
    url_base   = "https://www.oxfordreference.com/view/"

    try:
      opts, args = getopt.getopt(sys.argv[1:],"hf:o:t:")
    except getopt.GetoptError:
      help() 
      sys.exit(0)

    func = outb = topc = ""
    for opt, arg in opts:
        if opt == '-h':
            help()
            sys.exit(0)
        elif opt in ("-f"):
            func = arg     
        elif opt in ("-o"):
            outb = arg     
        elif opt in ("-t"):
            topc = arg                
    if func == "get_topic_htmls":
        get_topic_htmls(get_topic_param(topc), outb)
    elif func == "parse_htmls":
        parse_htmls(outb)
    else:
        print("Unknown function:",func)
        help()
        
        
        
        
        
        
