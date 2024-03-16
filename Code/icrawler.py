#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install icrawler')


# In[ ]:


from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(storage={'root_dir': '/content/drive/MyDrive/FYP_Degree/Yoga-82(Splited)/Yogic_sleep_pose'})
google_crawler.crawl(keyword='Yogic sleep pose', max_num=300)


# In[ ]:




