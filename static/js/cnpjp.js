function getCarsPosition(blob) {
    /**
     * Get vehicles position and size
     */

    /*
     * Usage:
        getCarsPosition(blob)
          .then(result => {
              console.log(result);
              callback(result.success, result.data)
          })
          .catch(error => {
              console.error(error);
          });
     */

    return new Promise((resolve, reject) => {
        const endpoint = "/cars/mark";

        let result = {
            success: false,
            message: null,
            data: []
        }

        const formData = new FormData();
        const fileName = `${uuidv4()}.jpg`;
        formData.append('file', blob, fileName);

        $.ajax({
            url: endpoint,
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                result.success = true;
                result.message = "Successfully.";
                result.data = response;
                resolve(result);
            },
            error: function (xhr) {
                result.message = xhr.statusText || "An error occurred";
                reject(result);
            }
        });
    });
}