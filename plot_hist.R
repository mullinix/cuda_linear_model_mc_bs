#!/usr/bin/env Rscript
## Plot Histograms!

# load plotting library and color themes
library(ggplot2)
tableau_colors=read.csv('tableau10_palette_html.csv',header=FALSE)
tableau_colors=sapply(tableau_colors, function(x) paste("#",x,sep=""))

plot_histogram=function(filename){
  # set file names from input
  hist_file=sprintf("%s-slope-histogram.dat",filename)
  percentile_file=sprintf("%s-slope-CI.dat",filename)
  pdf_file=sprintf("%s-histogram.pdf",filename)
  # prepare histogram
  histogramdata=read.csv(hist_file,header=FALSE)
  colnames(histogramdata)=c('breaks','counts')
  bin_size=mean(diff(histogramdata$breaks))
  slope=histogramdata$breaks-bin_size/2
  counts=histogramdata$counts
  frequency=counts/sum(counts)
  histogramdata=data.frame(slope,frequency)
  
  # load confidence interval data
  CIdata=read.csv(percentile_file,header=FALSE)
  colnames(CIdata)=c('lower_percentile','upper_percentile','lower_BCa','upper_BCa','lower_SEM', 'upper_SEM','middle','mean','npts','Nbs')
  
  # prepare legend
  legendlabels=c("95% CI (BCa)","SEM","Median","Mean")
  names(tableau_colors)=legendlabels
  
  # prepare plots of CIs and centrality
  linesize=0.5
  vlines=data.frame(values=c(CIdata$lower_BCa,CIdata$lower_SEM,CIdata$middle,CIdata$mean),Legend=legendlabels)
  
  # create ggplot object
  p=ggplot(histogramdata, aes(x=slope, y=frequency)) + 
    geom_bar(stat="identity",width=bin_size) +
    geom_vline(data=vlines, aes(xintercept=values, color=Legend), show.legend=TRUE, size=linesize)+
    geom_vline(xintercept = CIdata$upper_SEM,show.legend=FALSE,color=tableau_colors[2], size=linesize) + 
    geom_vline(xintercept = CIdata$upper_BCa,show.legend=FALSE,color=tableau_colors[1], size=linesize) +
    scale_color_manual("",values = tableau_colors)+
    theme_classic(base_family="Helvetica",base_size=9)+
    theme(text=element_text(size=9))
  
  # save histogram to file
  WidthInInches=8.75/2.54;
  ggsave(filename=pdf_file,device="pdf",plot=p,width=2*WidthInInches,height=WidthInInches,units="in",dpi=600);
}
