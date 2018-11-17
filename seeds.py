import pandas as pd
import numpy as np
import random 
import math
import matplotlib.pyplot as plt
import sys

def PANDAS():
	f = open("seeds.txt")
	line = [float (i)for i in f.readline().split()]
	seedsDF= pd.DataFrame(np.array(pd.Series(line)).reshape(-1,7))
	line = [float (i)for i in f.readline().split()]
	while line:
		seedsDF=seedsDF.append(pd.DataFrame(np.array(pd.Series(line)).reshape(-1,7)))
		line = [float (i)for i in f.readline().split()]
		

	return seedsDF
def ClosestSet(closestdistance, distancearray, SeedsDF):
	for i in range(len(closestdistance)):
		closestdistance[i]=[]
	for j in range(len(distancearray[0])):
		minval=999999
		for i in range(len(distancearray)):
			#print i, j
			#print i, j, distancearray[i][j]
			minval=min(distancearray[i][j], minval)
		for i in range(len(distancearray)):
			if (minval==distancearray[i][j]):
				i, j, len(distancearray[0])
				closestdistance[i].append(pd.DataFrame(np.array(pd.Series(SeedsDF.iloc[j])).reshape(-1,7)))
	return closestdistance

def SSECalc(closestdistance1):
	SSEFinal=[]
	for i in closestdistance1:
		compare=0
		for j in i:
			compare+= j.sum(axis=1)[0]
		compare=float(compare/len(i))
		SSE=0
		for j in i:
			SSE+=math.pow(j.sum(axis=1)[0]- compare,2)
		SSEFinal.append(SSE)
	sumTotal=0
	for i in SSEFinal:
		sumTotal+=i
	return sumTotal
def KMeanCalc(closestdistance1):
	MeanFinal=[]
	counter = 0
	for i in (closestdistance1):
		
		MeanFinal.append([])
		length=0
		for j in i[0]:
			length+=1
		for k in range(length):
			addition=0
			for j in range(len(i)):
				addition+=i[j][k][0]
			#print addition
			MeanFinal[counter].append(addition/len(i))
		counter+=1
	return MeanFinal


def SSE(Centroids):
	finalSSE=[]
	for periods in range(0,10):
		SeedsDF=PANDAS()
		ra=[]
		randarray=[]
		distancearray=[]
		distancearray1=[]
		closestdistance=[]
		Plots=[]
		#---------------Getting Random Postiions for centroids---------------------
		while len(randarray)<Centroids:
			ri =random.randint(0, len(SeedsDF)-1)
			if ri not in ra:
				ra.append(ri)
				randarray.append(pd.DataFrame(np.array(pd.Series(SeedsDF.iloc[ri])).reshape(-1,7)))
				distancearray.append([])
				closestdistance.append([])
				distancearray1.append([])
				Plots.append([])
		#------------Computing Euclidan distance pre array------------------------
		#print SeedsDF
		for i in range(len(randarray)):
			for j in range(len(SeedsDF)):
				compare = pd.DataFrame(np.array(pd.Series(SeedsDF.iloc[j])).reshape(-1,7))
				compare = compare.sub(randarray[i])
				compare= abs(compare.sum(axis=1)[0])
				compare=math.sqrt( compare )
				distancearray[i].append(compare)
				

		#-------Create the closest Disnace Arrays----------------------------------
		closestdistance1 = ClosestSet(closestdistance, distancearray, SeedsDF)
		meanDistance = KMeanCalc(closestdistance1)
		meanarray=[]
		for i in meanDistance:
			meanarray.append(pd.DataFrame(np.array(pd.Series(i)).reshape(-1,7)))
		SSE=SSECalc(closestdistance1)
		counter=0
		for i in range(0,100):
			prev = SSE
			distancearray=[]
			while len(distancearray)<Centroids:
				distancearray.append([])
			for j in range(len(meanarray)):
				for k in range(len(SeedsDF)):
					compare = pd.DataFrame(np.array(pd.Series(SeedsDF.iloc[k])).reshape(-1,7))
					compare = compare.sub(meanarray[j])
					compare= abs(compare.sum(axis=1)[0])
					compare=math.sqrt( compare )
					distancearray[j].append(compare)
			print meanarray
			#print distancearray
			closestdistance1 = ClosestSet(closestdistance, distancearray, SeedsDF)
			meanDistance = KMeanCalc(closestdistance1)
			meanarray=[]
			for i in meanDistance:
				meanarray.append(pd.DataFrame(np.array(pd.Series(i)).reshape(-1,7)))
			SSE=SSECalc(closestdistance1)
			#print prev, SSE
			if abs(prev-SSE)<.001:
				#print meanDistance
				#print SSE
				print prev, SSE
				return SSE
				#prev=meanDistance



			'''
			#print meanDistance
			for j in range(len(distancearray)):
				distancearray[j]=[]
				for k in range(len(SeedsDF)):
					compare = pd.DataFrame(np.array(pd.Series(SeedsDF.iloc[k])).reshape(-1,7))
					compare= abs(meanDistance[j]-compare.sum(axis=1)[0]/7)
					distancearray[j].append(compare)
			closestdistance1 = ClosestSet(closestdistance, distancearray, SeedsDF)
			meanDistance = KMeanCalc(closestdistance1)
			BreakCase=True
			for j in range(len(prev)):
				if abs(prev[j]-meanDistance[j])>0.001 :
					BreakCase=False
			SSE=SSECalc(closestdistance1)
			for j in range(len(Plots)):
				Plots[j].append(SSE[j])
			if(BreakCase):
				for j in range(len(Plots)):

					plt.plot(range(counter+1), Plots[j], label=j, c=np.random.rand(3,1), linewidth=3.0)
				plt.legend(loc='upper left')
				plt.xlabel("Iteration")
				plt.ylabel("Score")
				plt.title("K-Centroids = "+ str(Centroids))
				plt.close()
				#------THIS NEEDS TO BE SORTED!------
				finalSSE.append(sorted(SSE))
				for j in closestdistance1:
					print len(j)
				print '\n\n'
				break
			counter+=1
	sumTotal=[]
	for j in range(len(finalSSE[0])):
		sumTotal.append(0)
		for i in range(len(finalSSE)):
			sumTotal[j]+=finalSSE[i][j]
		sumTotal[j]=float(sumTotal[j]/len(finalSSE))
	return sumTotal	
	'''
		
def main():
	centroids=[3,5,7]
	for i in centroids:
		average=SSE(i)
		print "Number of K-Centroids", i,  "Average SSE:", average

				
if __name__=="__main__":
	main()

