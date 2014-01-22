import cimemetics


if __name__ == '__main__':
    graph=cimemetics.loadData(cimemetics.dolphins)
    print cimemetics.NMI(graph,(None,cimemetics.dgt),(None,cimemetics.dgt2))
        
        
