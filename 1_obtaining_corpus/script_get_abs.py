# Proj: Plant science conceptual history
# By  : Shin-Han Shiu
# On  : 3/8/21  
#
# Entrez database and UID:
# https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi
# https://www.ncbi.nlm.nih.gov/books/NBK25497/table/chapter2.T._entrez_unique_identifiers_ui/?report=objectonly
# 
# Entrez rettype, retmode for efetch
# https://www.ncbi.nlm.nih.gov/books/NBK25499/table/chapter4.T._valid_values_of__retmode_and/
#
#

import sys, getopt
#import entrezpy.conduit
import hashlib
import gzip
import re
import os
from pathlib import Path

class fetch_abs:
    """A class for rectrieving relevant pubmed records"""
    
    def get_viridiplantae(self):
        """Get the genus records for viridiplantae"""
        c    = entrezpy.conduit.Conduit('shius@msu.edu')
        pipe = c.new_pipeline()     

        # Parameter explained:
        # https://www.ncbi.nlm.nih.gov/books/NBK25499/
        sid  = pipe.add_search({'db'  : 'taxonomy', 
                               'term': 'txid33090[Subtree]'})
        fid  = pipe.add_fetch({'retmode' : 'xml',
                               'retmax'  : '10'}, 
                               dependency=sid)
        rec  = c.run(pipe)
        print('Done!')
        
    def get_genus_names(self, infile):
        """[NOT USED] Parse the downloaded XML from get_viridiplantae()"""
        inp    = open(infile)
        inl    = inp.readline()
        
        geni   = {} # genus dict: {scinm:[synonym]}
        scinm  = '' # scientific name
        synnm  = [] # synonym
        taxtag = '<Taxon>'
        scitag = '<ScientificName>'
        syntag = '<Synonym>'
        rnktag = '<Rank>'
        found_taxtag = 0
        found_rnktag = 0
        count  = 0
        clines = 0
        while inl != '':
            if inl.find(taxtag) != -1:
                found_taxtag = 1
                if count % 1e4 == 0:
                    print(f'{count/1e4} x 10k')
                count += 1
            elif found_taxtag and inl.find(scitag) != -1:
                bidx = inl.find(scitag) + len(scitag) # begin idx
                eidx = inl.find('</')                 # ending idx
                # In case it is a species name or has other stuff
                scinm = inl[bidx:eidx].split(' ')[0]
                #print('>',scinm)

                found_taxtag  = 1
                
                # new scinm, reset couple variables
                synnm        = [] # reset synonym list
                found_rnktag = 0  
                found_taxtag = 0  # to ensure only the first scinm is parsed
                #print(scinm)
            elif inl.find(syntag) != -1:
                bidx = inl.find(syntag) + len(syntag) # begin idx
                eidx = inl.find('</')
                synnm.append(inl[bidx:eidx].split(' ')[0])
                #print('>>',synnm)
            elif found_rnktag == 0 and inl.find(rnktag) != -1 and \
                 inl.find('genus') != -1:
                if scinm in geni:
                    #print('Redundant:',scinm, "line:",clines)
                    pass
                else:
                    geni[scinm] = synnm
                #print("<<ADD>>")
                # This is set to ignore all subsequent rank tags.
                found_rnktag = 1
                  
            inl = inp.readline()
            clines += 1
        
        # Generate output
        oup = open(f'{optd["-i"]}.genus_names', 'w')
        oup.write('Genus\tSynonyms\n')
        for i in geni:
            oup.write(f'{i}\t{geni[i]}\n')
        oup.close()
               
    def get_taxa_names(self, infile):
        """Get all taxa (rank) names from the lineage line for genus records"""
        
        rnk_tag = '    <Rank>'
        lin_tag = '    <Lineage>'
        rnkflag = 0
        linflag = 0
        inp     = open(infile)
        inl     = inp.readline()
        taxa_dc = {}
        c = 0
        while inl != "":
            # This record is ranked genus.
            if inl.startswith(rnk_tag) and inl.find("genus") != -1:
                rnkflag = 1
                if c % 1e4 == 0:
                    print(f'{c/1e4} x 10k')
                c += 1
                print("<genus>",[inl])
            elif inl.startswith(lin_tag) != -1 and rnkflag:
                taxa = inl.split('</')[0].split('; ')
                # put any thing below Viridiplantae into a dictionary
                qualified = 0
                print("  <lineage>")
                for rank in taxa:
                    if rank == "Viridiplantae":
                        qualified = 1
                    elif qualified:
                        taxa_dc[rank] = 0
                linflag = 1        
                rnkflag = 0
                    
            inl = inp.readline()
        
        oup = open(infile + ".all_taxa", "w")
        for i in taxa_dc:
            oup.write(f"{i}\n")
        oup.close()
        
    def check_md5(self, DIR):
        flist = os.listdir(DIR)
        # Get the compressed file names and corresponding md5 files.
        qfiles = []
        for i in flist:
            if i.startswith('pubmed21n') and not i.endswith('md5'):
                qfiles.append(i)
        
        c = 0
        for i in qfiles:
            #print(i)
            file_path = Path(DIR) / i
            md5  = self.get_md5(file_path)
            
            cksm_path = Path(DIR) / (i+".md5")
            chk  = open(cksm_path).readline().strip().split(" ")[-1]
            
            #print(md5,chk)
            countE = 0
            if md5 != chk:
                print("!!!checksum failed:",i)
                countE += 1
            if c % 10 == 0:
                print(f'{c/10} x 10')
            c += 1
        print(f"Checked {c}, failed {countE}")
    
    def get_md5(self, fname):
        """Based on: https://stackoverflow.com/questions/3431825"""
        hash_md5 = hashlib.md5()
        
        # Deviation: include gzip.open, see:
        # https://stackoverflow.com/questions/39651277
        # But found out that the checksum is for the zipped version.
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def batch_check_taxa_words(self, taxa_file, xml_dir, log_file):
        """Run check_taxa_words in batch"""
        
        all_files = os.listdir(xml_dir)
        all_in_dc = {}
        for i in all_files:
            if i.endswith(".xml.gz"):
                print(i)
                # Check if this has been run already
                qfile = Path(xml_dir) / (i+".qualified")
                if os.path.isfile(qfile):
                    print(" processed")
                else:
                    path = Path(xml_dir) / i
                    all_in_dc[i] = self.check_taxa_words(taxa_file, path)

        # Issue: This output may not be complete. If the run failed halfway
        # through, there will be no output. Subsequent run will only have
        # partial records. Not ideal.
        oup = open(log_file, "w")
        oup.write("PubMed file\tIn_JT\tIn_AT\tIn_AB\n")
        for i in all_in_dc:
            in_dc = all_in_dc[i]
            oup.write(f"{i}\t{in_dc[1]}\t{in_dc[2]}\t{in_dc[3]}\n")
            
        oup.close()
               
    def check_taxa_words(self, taxa_file, xml_file):
        """For each record, ask if any of the plant taxa word is present"""
        
        # Taxa word file
        taxa = {}
        for i in open(taxa_file).readlines():
            taxa[i.strip().lower()] = 0
        
        # Get the following info from XML
        # 0. <PMID Version="1">
        # 1. <Journal> --> <PubDate> --> <Month>, <Year>
        # 2. <Journal> --> <Title> lower cased
        # 3. <ArticleTitle> lower cased
        # 4. <AbstractText> lower cased
        
        pmid_tag  = "      <PMID"
        # 3/16/21 Change to PubDate insteadd
        #date_tag  = "      <DateCompleted>"
        #date_tag2 = "      </DateCompleted>"
        date_tag  = "            <PubDate>"
        date_tag2 = "            </PubDate>"
        year_tag  = "              <Year>"
        mont_tag  = "              <Month>"    # some entries don't have this
        jttl_tag  = "          <Title>"
        attl_tag  = "        <ArticleTitle>"
        abst_tag  = "          <AbstractText>"
        pmid_flag = 0
        date_flag = 0
        pubmed_dc = {}
        PMID = PubDate = YR = MO = JTitle = ATitle = Abstract = ""
         
        inp  = gzip.open(xml_file, 'rt')  # Input file is gzipped, read as text
        inl  = inp.readline()
        c = 0
        q = 0
        in_dc = {1:0, 2:0, 3:0}
        ERR_dc = {}
        MO_dc = {"Jan":"01", "Feb":"02", "Mar":"03", "Apr":"04",
                 "May":"05", "Jun":"06", "Jul":"07", "Aug":"08",
                 "Sep":"09", "Oct":"10", "Nov":"11", "Dec":"12"}
        while inl != "":
            if inl.startswith(pmid_tag):
                PMID = self.parse_xml(inl)
                pmid_flag = 1
                #print(PMID)
                #if c % 1e3 == 0:
                #    print(f" {c/1e3} x 1e3")
                c += 1
            elif pmid_flag:
                if inl.startswith(date_tag):
                    date_flag = 1
                elif inl.startswith(year_tag) and date_flag:
                    YR = self.parse_xml(inl)
                elif inl.startswith(mont_tag) and date_flag:
                    MO = self.parse_xml(inl)

                    # Mot all month format use abbrev, some use numbers
                    if MO in MO_dc:
                        MO = MO_dc[MO]
                elif inl.startswith(date_tag2):
                    date_flag = 0

                    # Month info not found
                    if MO == "":
                        ERR_dc[PMID] = "no_month_info"
                    PubDate = f"{YR}{MO}"
                    #print("\t",PubDate)
                elif inl.startswith(jttl_tag):
                    JTitle = self.parse_xml(inl).lower()
                    #print("\t",JTitle)
                elif inl.startswith(attl_tag):
                    ATitle = self.parse_xml(inl).lower()
                    #print("\t",ATitle)
                elif inl.startswith(abst_tag):
                    Abstract = self.parse_xml(inl).lower()
                    #print("\t",Abstract[:50])
                    pmid_flag = 0

                    [match_flag, in_flag] = self.match_taxa(
                                                taxa, JTitle, ATitle, Abstract)

                    # match plant words and pubdate has year and month
                    if match_flag and len(PubDate) == 6:
                        pubmed_dc[PMID] = [PubDate, JTitle, ATitle, Abstract]
                        q += 1
                        in_dc[in_flag] += 1
                        
            inl = inp.readline()
        print(f" total {c} records, {q} qualified," +
              f" {in_dc[1]} in JT, {in_dc[2]} in AT, {in_dc[3]} in AB")
        print(" # with no month info:",len(ERR_dc.keys()))
        for i in ERR_dc:
            print(" ",i)

        self.write_qualified(xml_file, pubmed_dc)
        
        return in_dc
        
    def write_qualified(self, xml_file, pubmed_dc):
        """Write the qualified pubmed entries into an XML file"""
        oup = open(str(xml_file) + ".qualified", "w")
        for PMID in pubmed_dc:
            [PD, JT, AT, AB] = pubmed_dc[PMID]
            oup.write(f"{PMID}\t{PD}\t{JT}\t{AT}\t{AB}\n")
        oup.close() 
        
    def match_taxa(self, taxa, JT, AT, AB):
        """Determine if any taxa tag, or plant is mentioned
        
        return
          Found (1) or not (0)
          In JT, AT, or AB first
        """
        
        # Check match in journal title
        # [\w] means any alphanumeric character and is equal to the 
        # character set [a-zA-Z0-9_] a to z, A to Z , 0 to 9 and underscore. 
        JT_list = re.sub("[^\w]", " ",  JT).split()
        if "plant" in JT_list or "plants" in JT_list:
            #print(" found in JT")
            return [1, 1]
            
        # Check article title
        AT_list = re.sub("[^\w]", " ",  AT).split()
        for i in AT_list:
            if i in taxa:
                #print(" found in AT")
                return [1, 2]

        # Check article abstract
        AB_list = re.sub("[^\w]", " ",  AB).split()
        for i in AB_list:
            if i in taxa:
                #print(" found in AB")
                return [1, 3]
        
        return [0, 0]
                        
    def parse_xml(self, xml):
        """Get the string between a pair of beginning and ending XML tags"""
        return xml[xml.find(">")+1:xml.rfind("<")]

    def combined_qualified(self, DIR):
        """Combined qualified pubmed records and rid of reudndant entries
        DIR: directory with all .qualified files after batch_check_taxa_words()
             runs.
        """
        qfiles = os.listdir(DIR)
        # Sort the q file in reverse order
        qfiles.sort(reverse=True)

        # Construct PMID dictionary and generate a log output
        oup = open('log_pubmed_qualified.err', 'w')
        PMIDs = {}
        print('Process:')
        count_TooNew = 0
        count_TooOld = 0
        for i in qfiles:
            print("",i)
            path = Path(DIR) / i
            # 3/18/20: deal with gzip rather than normal text file.
            inp  = gzip.open(path, 'rt')
            inl  = inp.readline()
            while inl != "":
                # pmid, pub date, journal title, article title, abstract
                [pmid, pd, jt, at, ab] = inl.split('\t')
                pd2 = int(pd)
                if pd2 >= 202101:
                    count_TooNew += 1
                elif pd2 < 196101:
                    count_TooOld += 1
                elif pmid not in PMIDs:
                    PMIDs[pmid] = [pd2, pmid, pd, jt, at, ab]
                else:
                    # More recent
                    if pd2 > PMIDs[pmid][0]:
                        oup.write(f'REDUN: {pmid}: {pd2} > {PMIDs[pmid][0]}\n')
                        PMIDs[pmid] = [pd2, pmid, pd, jt, at, ab]
                inl = inp.readline()
        oup.close()

        print("Write all qualified...")
        oup = open('pubmed_qualified', 'w')
        for pmid in PMIDs:
            oup.write('\t'.join(PMIDs[pmid][1:]))
        print(f' # entries:,{len(PMIDs.keys())}')
        print(f' excluding: {count_TooOld} too old, {count_TooNew} too new')
        oup.close()

    def help(self):
        print('Usage: python script_0_get_abs.py <options>')
        print(' -f function_name')
        print('     get_viridiplantae: get Vridiplantae taxa xmls. Need: -e')
        print('     get_genus_names: parse genus names from xmls. Need: -i')
        print('     get_taxa_names: parse all taxa names from xmls. Need: -i')
        print('     check_md5: of downloaded pubmed files. Need: -d')
        print('     check_taxa_words: in one pubmed file. Need:-x,-t')
        print('     batch_check_taxa_words: in all pubmed files. Need:-d,-t,-l')
        print('     combined_qualified: no redundant pmid. Need: -d')
        print(' -e email address for NCBI eutility')
        print(' -i general input file')
        print(' -d directory')
        print(' -t taxa file generated by get_taxa_names')
        print(' -x xml file with multiple records')
        print(' -l log file name')
        
#-------------------------------------------------------------------------------

if __name__ == "__main__":

    fa   = fetch_abs()
    argv = sys.argv[1:]
    
    opts, args = getopt.getopt(argv, 'f:i:d:t:x:l:e:')
    
    # Turn opts list into a dict
    optd = {}
    for i in opts:
        optd[i[0]] = i[1]

    # Run function
    func = optd['-f']
    if func == 'get_pubmed_viridiplantae':
        email = optd['-e']
        fa.get_pubmed_viridiplantae()
    elif func == 'get_genus_names':
        fa.get_genus_names(optd)
    elif func == 'get_taxa_names':
        infile = optd['-i']
        fa.get_taxa_names(infile)
    elif func == 'check_md5':
        DIR = optd['-d']  # target directory
        fa.check_md5(DIR)
    elif func == 'check_taxa_words':
        xml_file  = optd['-x']  # PubMed XML
        taxa_file = optd['-t']
        fa.check_taxa_words(taxa_file, xml_file)
    elif func == 'batch_check_taxa_words':
        xml_dir   = optd['-d']  # PubMed XML
        taxa_file = optd['-t']
        log_file  = optd['-l']
        fa.batch_check_taxa_words(taxa_file, xml_dir, log_file)
    elif func == 'combined_qualified':
        DIR = optd['-d']  # target directory
        fa.combined_qualified(DIR)
    else:
        print('Err: Function unknown:',func)
        fa.help()
