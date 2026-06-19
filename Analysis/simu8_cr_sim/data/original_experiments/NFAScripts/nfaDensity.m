%script nfaDensity
  %calculates the average distance between faces in a given study set 
  %of faces, then computes the average average for a list.  This is to 
  %determine what constitutes a denser or more spread out study list.
  %Then, we go through all the subjects lists and see which ones are
  %below average density and which ones are above average density, and 
  %do our neighborhood analyses for both conditions.  Then we have to
  %compare the slopes/intercepts for both conditions: What is the
  %appropriate intercept for that?

  function [list1,list2] = nfaDensity(events)
     half1=filterStruct(events,'halfnum==1');
        dim1=[];
        dim2=[];
        dim3=[];
        dim4=[];
     for face1=1:8
        dim1=[dim1; half1(face1).facecoordinates(1)];
        dim2=[dim2; half1(face1).facecoordinates(2)];
        dim3=[dim3; half1(face1).facecoordinates(3)];
        dim4=[dim4; half1(face1).facecoordinates(4)]; 
     end
     center=[mean(dim1) mean(dim2) mean(dim3) mean(dim4)];
     ssd=[];
     for face1=1:8
         ssd=[ssd; (EDist(half1(face1).facecoordinates,center))^2];
     end
     ssd=sum(ssd);
     squaredd=ssd/8;
     list1=sqrt(squaredd);   
     
     half2=filterStruct(events,'halfnum==2');
	dim1=[];
        dim2=[];
        dim3=[];
        dim4=[];
     for face1=1:8
	dim1=[dim1; half2(face1).facecoordinates(1)];
        dim2=[dim2; half2(face1).facecoordinates(2)];
        dim3=[dim3; half2(face1).facecoordinates(3)];
        dim4=[dim4; half2(face1).facecoordinates(4)];
     end
     center=[mean(dim1) mean(dim2) mean(dim3) mean(dim4)];
     ssd=[];	
     for face1=1:8
	ssd=[ssd; (EDist(half2(face1).facecoordinates,center))^2];
     end
     ssd=sum(ssd);
     squaredd=ssd/8;
     list2=sqrt(squaredd);
  end
    

	       

    
