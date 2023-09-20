function testMatFormat(data)
    arguments
        data (4,4) cell
    end
    cols = {'white','red','green','blue','yellow'};
    for p=1:16
        if ~matches(data{p},cols)
            warning('Fail: Unknown colour %s',data{p})
        end
    end
    fprintf('Passed\n')