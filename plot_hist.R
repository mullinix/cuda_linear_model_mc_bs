#!/usr/bin/env Rscript
## Plot Histograms!

# load plotting library and color themes
library(ggplot2)
tableau_colors=read.csv('tableau10_palette_html.csv',header=FALSE)
tableau_colors=sapply(tableau_colors, function(x) paste("#",x,sep=""))

plot_histogram=function(filename){
  # load confidence interval data
  percentile_file=sprintf("%s-CI.dat",filename)
  CIdata=read.csv(percentile_file,header=TRUE)
  
  # set file names from input
  for(statistic in CIdata$statistic){
    hist_file=sprintf("%s-%s-histogram.dat",filename,statistic)
    pdf_file=sprintf("%s-%s-histogram.pdf",filename,statistic)
    # prepare histogram
    histogramdata=read.csv(hist_file,header=FALSE)
    colnames(histogramdata)=c('breaks','counts')
    bin_size=mean(diff(histogramdata$breaks))
    stat_data=histogramdata$breaks-bin_size/2
    counts=histogramdata$counts
    frequency=counts/sum(counts)
    histogramdata=data.frame(stat_data,frequency)
    
    # prepare legend
    legendlabels=c("95% CI (BCa)","Mean","Median","SEM")
    names(tableau_colors)=legendlabels
    
    # prepare plots of CIs and centrality
    CIs=CIdata[CIdata$statistic==statistic,]
    linesize=0.5
    vlines=data.frame(values=c(CIs$lower_BCa,CIs$mean,CIs$median,CIs$lower_SE),Legend=legendlabels)
    
    # create ggplot object
    p=ggplot(histogramdata, aes(x=stat_data, y=frequency)) + 
      geom_bar(stat="identity",width=bin_size) +
      geom_vline(data=vlines, aes(xintercept=values, color=Legend), show.legend=TRUE, size=linesize) +
      geom_vline(xintercept = CIs$upper_BCa,show.legend=FALSE,color=tableau_colors[1], size=linesize) + 
      geom_vline(xintercept = CIs$upper_SE,show.legend=FALSE,color=tableau_colors[4], size=linesize) +
      scale_color_manual("",values = tableau_colors) +
      theme_classic(base_family="Helvetica",base_size=9) +
      theme(text=element_text(size=9)) +
      labs(x = statistic)
    
    # save histogram to file
    WidthInInches=8.75/2.54;
    ggsave(filename=pdf_file,device="pdf",plot=p,width=2*WidthInInches,height=WidthInInches,units="in",dpi=600);
  }
}
