function pattern_search_training(syn_path, gt_path, photo_path, henney_path)
%Solving using pattern search%
FUN = {@demo_solar, syn_path, gt_path, photo_path,henney_path, 0};
A = [];
b = [];
Aeq = [];
beq = [];
LB = [0.2 -3];
UB = [1 3];
X0 = [0.5 0];

opts = optimoptions('patternsearch','Display','iter','PlotFcn',@psplotbestf);
[X1,Fval,Exitflag,Output]=patternsearch(FUN,X0,A,b,Aeq,beq,LB,UB,opts);
%writing result to results.txt %
ftxt = fopen('results_training.txt','at');
fprintf(ftxt,sprintf('\n'));
fprintf(ftxt,'%s\t',syn_path);
fprintf(ftxt,'%f\t',X1(1));
fprintf(ftxt,'%f\t',X1(2));
%fprintf(ftxt,'%f\t',X1(3));
fprintf(ftxt,'%f\t',Fval);
fclose(ftxt);
end