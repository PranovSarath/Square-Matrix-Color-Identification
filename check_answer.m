function max_match = check_answer(answer, filename)
arguments
    answer (4,4) cell
    filename string
end
% load the actual answer
load(filename,'res')

%check it is correctly formated. 
testMatFormat(answer)



% check

res2=res;

matches{1} = cellfun(@strcmp,answer,res2);

%rot 
res2 = rot90(res2);
matches{2} = cellfun(@strcmp,answer,res2);

res2 = rot90(res2);
matches{3} = cellfun(@strcmp,answer,res2);

res2 = rot90(res2);
matches{4} = cellfun(@strcmp,answer,res2);

% now do the flipped version
res2=fliplr(res);

matches{5} = cellfun(@strcmp,answer,res2);

%rot 
res2 = rot90(res2);
matches{6} = cellfun(@strcmp,answer,res2);

res2 = rot90(res2);
matches{7} = cellfun(@strcmp,answer,res2);

res2 = rot90(res2);
matches{8} = cellfun(@strcmp,answer,res2);


sum2= @(x) sum(x,'all');
best_match = cellfun(sum2,matches)/16*100;
%[max_match, idx] = max(best_match);
max_match = max(best_match);
max_match
% get the actual answer that matches the provided answer
%if idx <= 4 % rotation matches
%    actual_answer = res;
%    for i = 1:idx
%        actual_answer = rot90(actual_answer);
%    end
%else % flipped version matches
%    actual_answer = fliplr(res);
%    for i = 1:(idx-4)
%        actual_answer = rot90(actual_answer);
%    end
%end
%actual_answer
