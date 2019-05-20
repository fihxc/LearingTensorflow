#include <bits/stdc++.h>
using namespace std;

int main()
{
    // data.txt
    freopen("in.txt", "r", stdin);
    for(int i=0; i < 73; i++)
    {
        for(int j =0;j<324;j++)
        {
            int num = 0;
            if(j<303) scanf("%d,", &num);
            printf("%4d", num);
        }
        cout << endl;
    }

    // tag.txt
    // freopen("res.txt", "r", stdin);
    // for(int i=0; i < 73; i++)
    // {
    //     int num = 0;
    //     scanf("%d,", &num);
    //     for(int j =0; j<20;j++)
    //     {
    //         if (j == num) printf("1");
    //         else printf("0");
    //         if(j==19) printf("\n");
    //         else printf(" ");
    //     }
    // }
    return 0;
}