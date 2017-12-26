        identity_clus_name=[]
        identity_clus_feat=[]
        centroids=[]
        for cluster_id in set(clusters):
         i=0
         cluster_features=[]
         cluster_name_list=[]
         for stf in clusters:

                if stf==cluster_id:

                        cluster_features.append(all_feat[i])
                        id_list.append(cluster_id)
                        cluster_name_list.append(all_label[i])

                i+=1
         cluster_features = np.asarray(cluster_features)
         '''
         number of images in the cluster should be > 1. This implies that within_dist doesn't return error.

         what if all clusters have 1 image? Poor clustering!! Declare the other method as winner

         However if the number is 2, the within distance will have only one distance value, causing variance to be 0.
         We can later remove cases where within variance is 0.

         but what if every cluster has 2 images?
         in that case, it would mean that the 2 samples are exceptionally similar, hence the distance should be very low... instead of minimizing var
         its intuitive to minimize this distance


         '''

         centroid = sum(cluster_features)/len(cluster_features)
         centroids.append(centroid)
         identity_clus_feat.append(cluster_features)
         identity_clus_name.append(cluster_name_list)


        centroids=np.asarray(centroids)
        sim_centroids=99999
        x=0
        for c in centroids:
                y=0
                for d in centroids:
                        if sim_centroids >  abs(1-cosine(c,d)):
                                sim_centroids=abs(1-cosine(c,d))
                                id_c = x
                                id_d = y
                        y+=1
                x+=1
        print "THESE IMAGES ARE MOST DISSIMILIAR"
        print identity_clus_name[id_c]
        print "WITH THESE IMAGES"
        print identity_clus_name[id_d]
        print "BECAUSE SIMILARITY  BW THEIR RESPECTIVE CLUSTERS IS"
        print sim_centroids
        for c in identity_clus_feat[id_c]:
                for d in identity_clus_feat[id_d]:
                        print "SIM BETWEEN", "and",
                        print "IS", abs(1-cosine(c,d))






def score(bet_var, within_var):

        c_score = bet_var/(within_var + 0.5)

        return c_score



def feat_extract(record):



        field = 'DEEPFEATURE_'
        img_feat = []
        i=1
        while i<=512 :
                img_feat.append(float(record[field+str(i)]))
                i+=1
        return img_feat
def face_detect(name):

        record = (df[df['FILE']==name])
        x= record['FACE_X']
        y= record['FACE_Y']
        h= record['FACE_HEIGHT']
        w= record['FACE_WIDTH']
        return float(x),float(y),float(h),float(w)

#improve csv read
#write eval func --lda criterion
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

def cos_sim(a, b):
        """Takes 2 vectors a, b and returns the cosine similarity according
        to the definition of the dot product
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)
def visualise(best_images,savename):
        orig_images = []
        for im_name in best_images:
                x,y,h,w = face_detect('/'.join(im_name.split('/')[-2:]))

                img_data = Image.open(im_name)
                img2 = img_data.crop((int(x), int(y), int(x+h),int(y+w) ))
                orig_images.append(img2)
        images=[]
        for a in orig_images:
                images.append(a.resize((128,128)))
        draw = ImageDraw.Draw(images[-1])
        font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 25)
        draw.text((20, 0),"Probe", (255,0,0),font=font)
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
          new_im.paste(im, (x_offset,0))
          x_offset += im.size[0]

        new_im.show()
        new_im.save(savename+'.jpg')

def probe_based_dis(all_feat, probe_feat):
        my_dis=[]
        p = 0

        while p < all_feat.shape[0]:
                q=p+1
                while q < all_feat.shape[0] :
                        d0 = abs(cosine(all_feat[p], all_feat[q]))
                        d1 = abs(cosine(all_feat[p], probe_feat))
                        d2 = abs(cosine(all_feat[q], probe_feat))
                        s1 =abs(cos_sim(all_feat[p], probe_feat)) #similarity with probe
                        s2 =abs(cos_sim(all_feat[q], probe_feat))
                        #d1 = 1-p'x1, d2=1-p'x2 => d1+d2 = 2-p'(x1+x2)
                        d = math.sqrt(1/(s1*s2))
                        my_dis.append(d*d0)
                        q+=1
                p+=1
        return np.asarray(my_dis)


def nearest_neighbour(all_feat, probe_feat, all_label):
        dis = []
        name = []
        for feature,f_name in zip(all_feat,all_label):
                dis.append(abs(cosine(feature, probe_feat)))
                name.append(f_name)
        thresh = sum(dis)/len(dis)
        print "threshold : ", thresh

        name = np.asarray(name)
        dis = np.asarray(dis)

        name= name[dis<=thresh]
        dis= dis[dis<=thresh]
        visualise(name,'nn')

def evaluate(dis_mat, all_feat, all_label,savename):
        Z = linkage(dis_mat, 'average')
        e_value = dis_thresh(Z)
        get_cluster_stat(Z, e_value,all_feat, all_label)




req_sub_id = 0
my_dis_better = 0
total_instances=0
ids= os.listdir('/fs/janus-scratch/ankan/umdfaces/umdfaces_batch1/')
print ids
print len(ids)

for sub_id in ids:
 if os.path.isdir('/fs/janus-scratch/ankan/umdfaces/umdfaces_batch1/'+sub_id):
        all_feat=[]
        all_label=[]
        index=0
        print sub_id
        my_dis_better = 0
total_instances=0
ids= os.listdir('/fs/janus-scratch/ankan/umdfaces/umdfaces_batch1/')
print ids
print len(ids)

for sub_id in ids:
 if os.path.isdir('/fs/janus-scratch/ankan/umdfaces/umdfaces_batch1/'+sub_id):
        all_feat=[]
        all_label=[]
        index=0
        print sub_id
        imgs = os.listdir('/fs/janus-scratch/ankan/umdfaces/umdfaces_batch1/'+sub_id)
        #print imgs
        #print "------------------"
        for iden_img in imgs:
                #print iden_img
                #print "FILE...........", sub_id+'/'+iden_img
                record = (df_feat[df_feat['FILE']==sub_id+'/'+iden_img])
                if record.empty :

                        continue #check if file is in csv
                feature = feat_extract(record)
                all_feat.append(np.asarray(feature))
                all_label.append(sub_id+'/'+iden_img)


        if all_feat==[]:
                print "NOT FOUND"
                print imgs
                #ggg=raw_input()
                continue
        all_feat=np.asarray(all_feat)
        all_label=np.asarray(all_label)
        print all_feat.shape
        cos_dis = pdist(all_feat,'cosine')

        evaluate(cos_dis,all_feat,all_label,"cosine_")



print 'results:', my_dis_better, total_instances, my_dis_better/total_instances

                                                                                                                                                      294,0-1       Bot

                                                                                                                                                      263,1-8       87%
                        
                                                                                                                                                      220,1-8       70%
        
                                                                                                                                                      176,1-8       52%
        

                                                                                                                                                      133,0-1       35%
                                                                                                                                                      90,1-8        18%
